# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import torch
import torch.nn.functional as F
import cuda.tile as ct
from cuda.tile._backend import xpu

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd,
                   eps: ConstFloat, TILE_N: ConstInt):
    """
    Forward pass: computes mean/var, normalizes input, and applies affine transform.

    Args:
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        B: Bias tensor (N,).
        Y: Output tensor (M, N).
        Mean: Broadcast output mean tensor (M, TILE_N).
        Rstd: Broadcast output reciprocal standard deviation tensor (M, TILE_N).
        eps: Epsilon for numerical stability.
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute mean
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(
        Mean,
        index=(bid_m, 0),
        tile=ct.broadcast_to(mean, (1, TILE_N)),
    )

    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute variance
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        centered_tx = tx - mean
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(
        Rstd,
        index=(bid_m, 0),
        tile=ct.broadcast_to(rstd, (1, TILE_N)),
    )

    for j in range(num_tiles):
        # Normalize and apply affine transformation
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


def bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N):
    """Helper to load data and compute common backward terms."""
    tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
    tdy = ct.load(DY, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    xhat = (tx - mean) * rstd
    wdy = tw * tdy
    return tdy, xhat, wdy


@ct.kernel
def layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd,
                                   TILE_N: ConstInt):
    """
    Backward pass part 1: computes dX and partial dW/dB.
    Stores one partial dW/dB row per input row for final reduction.

    Args:
        DX: Output gradient with respect to X (M, N).
        DY: Input gradient with respect to Y (M, N).
        DW: Partial gradient with respect to W (M, N).
        DB: Partial gradient with respect to B (M, N).
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        Mean: Broadcast mean tensor (M, TILE_N).
        Rstd: Broadcast reciprocal standard deviation tensor (M, TILE_N).
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    mean = ct.load(Mean, index=(bid_m, 0), shape=(1, TILE_N))
    rstd = ct.load(Rstd, index=(bid_m, 0), shape=(1, TILE_N))

    c1 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute reduction terms for dX
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 += xhat * wdy
        c2 += wdy
    c1 = ct.sum(c1, axis=1) / N
    c2 = ct.sum(c2, axis=1) / N

    for j in range(num_tiles):
        # Compute dX and partial dW, dB
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy - (xhat * c1 + c2)) * rstd
        ct.store(DX, index=(bid_m, j), tile=tdx.astype(DX.dtype))

        partial_dw = (tdy * xhat).astype(DW.dtype)
        partial_db = tdy.astype(DB.dtype)

        # XPU: keep one partial row per input row. The CUDA sample folds rows
        # into 64 groups under locks, but CAS/spin loops are not yet supported
        # by tileir-to-mlir. The second kernel already performs this reduction.
        ct.store(DW, index=(bid_m, j), tile=partial_dw)
        ct.store(DB, index=(bid_m, j), tile=partial_db)


@ct.kernel
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    Backward pass part 2: Final reduction for dW and dB.

    Args:
        DW: Partial gradient with respect to W (TILE_M, N).
        DB: Partial gradient with respect to B (TILE_M, N).
        FINAL_DW: Replicated final gradient with respect to W (TILE_M, N).
        FINAL_DB: Replicated final gradient with respect to B (TILE_M, N).
        TILE_M: Number of partial gradients to reduce.
        TILE_N: Tile size along N dimension.
    """
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))

    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        # Sum partial gradients
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)

    final_dw = ct.broadcast_to(sum_dw[None, :], (TILE_M, TILE_N))
    final_db = ct.broadcast_to(sum_db[None, :], (TILE_M, TILE_N))
    ct.store(FINAL_DW, index=(0, bid_n), tile=final_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(0, bid_n), tile=final_db.astype(FINAL_DB.dtype))


# --- cuTile LayerNorm Wrapper ------------------------------------------------------

class CuTileLayerNorm(torch.autograd.Function):
    """
    A PyTorch Autograd Function wrapper for the cuTile LayerNorm kernel.
    This class manages the forward and backward passes, bridging PyTorch tensors
    with the cuTile kernel launches.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """
        Forward pass for LayerNorm.

        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor (*, ..., N).
            weight: Scale parameter (N,).
            bias: Shift parameter (N,).
            eps: Epsilon for numerical stability.

        Returns:
            Output tensor with the same shape as input.
        """
        # Flatten input to (M, N)
        x = input.reshape(-1, input.shape[-1])
        y = torch.empty_like(x)
        M, _ = x.shape
        TILE_N = 1024
        if x.shape[1] % TILE_N != 0:
            raise ValueError(
                f"XPU LayerNorm requires N divisible by {TILE_N}; got {x.shape[1]}"
            )

        # Allocate temporary buffers for mean and reciprocal standard deviation
        mean = torch.empty((M, TILE_N), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, TILE_N), dtype=torch.float32, device=x.device)
        # Launch the forward kernel with a 1D grid (M blocks)
        with xpu.compile_options({"wg_m": 1, "wg_n": TILE_N}):
            ct.launch(torch.xpu.current_stream(), (M, 1, 1), layer_norm_fwd,
                      (x, weight, bias, y, mean, rstd, eps, TILE_N))

        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.TILE_N = TILE_N

        return y.reshape(*input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LayerNorm.

        Computes gradients for input, weight, and bias using two kernels:
        1. Computes dX and partial reductions for dW and dB.
        2. Performs the final reduction for dW and dB.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient tensor of loss w.r.t. output (*, ..., N).

        Returns:
            Gradients for input, weight, bias, and None for eps.
        """
        x, weight, bias, mean, rstd = ctx.saved_tensors
        TILE_N = ctx.TILE_N
        M, N = x.shape

        # Flatten gradient output to (M, N)
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(dy)

        # XPU: retain one partial gradient per input row; the final kernel
        # reduces these rows without requiring lock/CAS support.
        dw = torch.empty((M, N), dtype=torch.float32, device=weight.device)
        db = torch.empty((M, N), dtype=torch.float32, device=bias.device)

        # Launch the first backward kernel to compute dX and partial dW/dB
        with xpu.compile_options({"wg_m": 1, "wg_n": TILE_N}):
            ct.launch(torch.xpu.current_stream(), (M, 1, 1), layer_norm_bwd_dx_partial_dwdb,
                      (dx, dy, dw, db, x, weight, mean, rstd, TILE_N))

        TILE_M = 32
        final_dw = torch.empty((TILE_M, N), dtype=weight.dtype, device=weight.device)
        final_db = torch.empty((TILE_M, N), dtype=bias.dtype, device=bias.device)

        # Launch the second backward kernel to reduce partial dW/dB
        with xpu.compile_options({"wg_m": TILE_M, "wg_n": TILE_N}):
            ct.launch(torch.xpu.current_stream(), (math.ceil(N / TILE_N), 1, 1), layer_norm_bwd_dwdb,
                      (dw, db, final_dw, final_db, TILE_M, TILE_N))

        return (dx.reshape(*grad_output.shape), final_dw[0], final_db[0], None)


def cutile_layer_norm(x, weight, bias, eps):
    return CuTileLayerNorm.apply(x, weight, bias, eps)


# --- Torch Reference Implementation -----------------------------------------

def torch_layer_norm(x, weight, bias, eps):
    return F.layer_norm(x, weight.shape, weight, bias, eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the cuTile LayerNorm output against a torch reference.",
    )
    args = parser.parse_args()

    print("--- Running cuTile LayerNorm Forward/Backward Sample (XPU) ---")

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available. Ensure Intel GPU runtime and PyTorch XPU are installed."
        )

    shape = (1024, 2048)
    dtype = torch.float32
    device = "xpu"
    weight = torch.randn(shape[-1], dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(shape[-1], dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    eps = 1e-5

    print(f"Input shape: {shape}, dtype: {dtype}, eps: {eps}")

    atol, rtol = 1e-2, 1e-2

    print("\n--- Executing cuTile LayerNorm Forward ---")
    y = cutile_layer_norm(x, weight, bias, eps)\

    print("\n--- Executing cuTile LayerNorm Backward ---")
    y.backward(dy, retain_graph=True)
    dx, dw, db = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    if args.correctness_check:
        print("\n--- Running correctness check against torch reference ---")
        y_ref = torch_layer_norm(x, weight, bias, eps)
        torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

        y_ref.backward(dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
        torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(dw, dw_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(db, db_ref, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled (use --correctness-check to enable)")

    print("\n--- cuTile LayerNorm Sample complete ---")
