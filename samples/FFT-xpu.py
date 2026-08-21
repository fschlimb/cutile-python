# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import cuda.tile as ct
import torch
from cuda.tile._backend import xpu

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")

# The tile program and factor generation are backend-independent. Keep them shared
# with the CUDA sample so the numerical algorithm cannot diverge between backends.
from FFT import fft_kernel, make_twiddles


def cutile_fft(
    x: torch.Tensor,
    factors: tuple,  # (F0, F1, F2) - factors of N
    atom_packing_dim: int = 64  # The 'D' parameter for data packing/unpacking
) -> torch.Tensor:
    """
    Performs a Batched 1D Fast Fourier Transform (FFT) using a cuTile kernel
    based on multi-dimensional factorization (similar to a Cooley-Tukey algorithm).

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, N) containing complex64 numbers.
                          This tensor must be on an XPU device.
        factors (tuple): A tuple (F0, F1, F2) representing the factors of N.
        atom_packing_dim (int): The dimension used for data packing/unpacking.

    Returns:
        torch.Tensor: Output tensor of shape (Batch, N) containing the FFT results.

    Raises:
        ValueError: If input tensor dimensions, device, or data type are incorrect,
                    if the provided factors do not multiply to N, or if N*2 is not
                    divisible by atom_packing_dim.
    """
    if x.ndim != 2:
        raise ValueError("Input tensor must be 2D (Batch, N).")
    if not x.is_xpu:
        raise ValueError("Input tensor must be on an XPU device.")
    if x.dtype != torch.complex64:
        raise ValueError("Input tensor dtype must be torch.complex64.")

    BS, N = x.shape
    F0, F1, F2 = factors
    if F0 * F1 * F2 != N:
        raise ValueError(f"Factors ({F0}*{F1}*{F2}={F0*F1*F2}) do not multiply to N={N}. "
                         f"Please provide factors that correctly decompose N.")
    if (N * 2) % atom_packing_dim != 0:
        raise ValueError(f"Total real/imag elements (N*2 = {N*2}) must be divisible by "
                         f"atom_packing_dim ({atom_packing_dim}) for kernel packing.")

    precision_dtype = x.real.dtype
    x_packed_in = torch.view_as_real(x).reshape(
        BS, N * 2 // atom_packing_dim, atom_packing_dim
    ).contiguous()
    W0_gmem, W1_gmem, W2_gmem, T0_gmem, T1_gmem = make_twiddles(
        factors, precision_dtype, x.device
    )
    y_packed_out = torch.empty_like(x_packed_in)

    # One workgroup handles one FFT batch entry. The XPU backend requires explicit
    # work-group tiles and a launch block for this non-MMA tiled kernel.
    grid = (BS, 1, 1)
    options = {
        "wg_m": BS,
        "wg_n": N,
        "block_threads": (1, 32, 16),
        "assume_in_bounds": True,
    }
    with xpu.compile_options(options):
        ct.launch(torch.xpu.current_stream(), grid, fft_kernel,
                  (x_packed_in, y_packed_out,
                   W0_gmem, W1_gmem, W2_gmem,
                   T0_gmem, T1_gmem,
                   N, F0, F1, F2, BS, atom_packing_dim))

    return torch.view_as_complex(y_packed_out.reshape(BS, N, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available. Ensure Intel GPU runtime and PyTorch XPU are installed."
        )

    print("--- Running cuTile FFT Example (XPU) ---")

    # Total FFT size (N) must be factorable into factors[0] * factors[1] * factors[2].
    BATCH_SIZE = 2
    FFT_SIZE = 8
    FFT_FACTORS = (2, 2, 2)
    ATOM_PACKING_DIM = 2

    torch.manual_seed(0)
    input_data_complex = torch.randn(
        BATCH_SIZE, FFT_SIZE, dtype=torch.complex64, device="xpu"
    )

    print("  Configuration:")
    print(f"  FFT Size (N): {FFT_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  FFT Factors (F0,F1,F2): {FFT_FACTORS}")
    print(f"  Atom Packing Dimension (D): {ATOM_PACKING_DIM}")
    print(f"Input data shape: {input_data_complex.shape}, dtype: {input_data_complex.dtype}")

    output_fft_cutile = cutile_fft(
        x=input_data_complex,
        factors=FFT_FACTORS,
        atom_packing_dim=ATOM_PACKING_DIM,
    )
    print(f"\ncuTile FFT Output shape: {output_fft_cutile.shape}, "
          f"dtype: {output_fft_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(output_fft_cutile, torch.fft.fft(input_data_complex, dim=-1))
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile FFT example execution complete ---")