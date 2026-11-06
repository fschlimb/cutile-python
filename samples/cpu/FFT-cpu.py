# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import cuda.tile as ct
import torch

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from FFT import fft_kernel, make_twiddles  # noqa: E402


ct.set_backend("cpu")


# --- Wrapper function to launch the fft_kernel ---
def cutile_fft(
    x: torch.Tensor,
    factors: tuple,  # (F0, F1, F2) - factors of N
    atom_packing_dim: int = 64,  # The 'D' parameter for data packing/unpacking
    num_cpu_threads: int = 0,
) -> torch.Tensor:
    """
    Performs a Batched 1D Fast Fourier Transform (FFT) using a cuTile kernel
    based on multi-dimensional factorization (similar to a Cooley-Tukey algorithm).

    This function prepares the input data, generates necessary pre-computed
    matrices, launches the cuTile kernel, and unpacks the results.

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, N) containing complex64 numbers.
                          This tensor *must* be on the CPU.
                          N (the FFT size) must be factorable into factors[0]*factors[1]*factors[2].
        factors (tuple): A tuple (F0, F1, F2) representing the factors of N.
                         The product F0 * F1 * F2 must equal N. These factors define
                         the logical 3D shape for the FFT decomposition within the kernel.
        atom_packing_dim (int): The dimension 'D' used for data packing/unpacking
                                in the kernel. This value affects memory access patterns.
                                The total number of real/imaginary elements (N*2) must be
                                divisible by this dimension. Default is 64.

    Returns:
        torch.Tensor: Output tensor of shape (Batch, N) containing the FFT results.
                      The output data type will be torch.complex64.

    Raises:
        ValueError: If input tensor dimensions, device, or data type are incorrect,
                    if the provided factors do not multiply to N, or if N*2 is not
                    divisible by atom_packing_dim.
    """
    # --- Input Validation ---
    if x.ndim != 2:
        raise ValueError("Input tensor must be 2D (Batch, N).")
    if not x.is_cpu:
        raise ValueError("Input tensor must be on the CPU.")
    if x.dtype != torch.complex64:
        raise ValueError("Input tensor dtype must be torch.complex64.")

    BS = x.shape[0]  # Extract Batch Size from the input tensor's shape.
    N = x.shape[1]   # Extract Total FFT size from the input tensor's shape.

    F0, F1, F2 = factors
    # Validate that the provided factors correctly decompose the total FFT size N.
    if F0 * F1 * F2 != N:
        raise ValueError(f"Factors ({F0}*{F1}*{F2}={F0*F1*F2}) do not multiply to N={N}. "
                         f"Please provide factors that correctly decompose N.")

    # Determine the underlying floating-point precision (e.g., float32) for
    # the real and imaginary parts of the complex numbers.
    PRECISION_DTYPE = x.real.dtype

    # --- Prepare Input Data for Kernel (Split real/imag, pack) ---
    # Convert the complex input tensor (BS, N) to a real tensor (BS, N, 2)
    # where the last dimension explicitly separates real and imaginary parts.
    x_ri = torch.view_as_real(x)

    # Reshape the real/imaginary tensor to the packed format (BS, N*2 // D, D)
    # that the kernel expects for efficient memory access.
    # This step assumes that the total number of real/imaginary elements (N*2)
    # is perfectly divisible by the `atom_packing_dim` (D).
    if (N * 2) % atom_packing_dim != 0:
        raise ValueError(f"Total real/imag elements (N*2 = {N*2}) must be divisible by "
                         f"atom_packing_dim ({atom_packing_dim}) for kernel packing.")
    x_packed_in = x_ri.reshape(BS, N * 2 // atom_packing_dim, atom_packing_dim).contiguous()

    # --- Generate W (Rotation) and T (Twiddle) Matrices ---
    # These matrices are pre-computed mathematically based on the FFT decomposition.
    # They are generated on the same device as the input tensor.
    W0_gmem, W1_gmem, W2_gmem, T0_gmem, T1_gmem = make_twiddles(factors, PRECISION_DTYPE, x.device)

    # --- Create Output Tensor ---
    # Initialize an empty tensor with the same shape and properties as the packed input.
    # This tensor will store the results computed by the kernel.
    y_packed_out = torch.empty_like(x_packed_in)

    # --- Calculate Grid Dimensions ---
    # For this FFT kernel, one thread block is launched for each item in the batch.
    # The grid is a 3-tuple (grid_x, grid_y, grid_z).
    grid = (BS, 1, 1)

    # --- Launch the cuTile Kernel ---
    # The `fft_kernel` is launched on the CPU with the calculated grid dimensions.
    # All necessary input tensors (packed data, W and T matrices) and constant parameters
    # (N, F0, F1, F2, BS, D) are passed to the kernel.
    with ct.backend_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(None, grid, fft_kernel,
                  (x_packed_in, y_packed_out,
                   W0_gmem, W1_gmem, W2_gmem,
                   T0_gmem, T1_gmem,
                   N, F0, F1, F2, BS, atom_packing_dim))

    # --- Unpack Output from Kernel (Reshape, combine real/imag) ---
    # Reshape the packed output tensor back to (BS, N, 2) to separate real/imaginary parts.
    y_ri = y_packed_out.reshape(BS, N, 2)
    # Convert the real/imaginary pair tensor back to a complex tensor (torch.complex64).
    y_complex = torch.view_as_complex(y_ri)

    return y_complex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()
    print("--- Running cuTile FFT Example ---")

    # --- User Configuration ---
    BATCH_SIZE = 2
    # Total FFT size (N) must be factorable into factors[0] * factors[1] * factors[2].
    # For example, N = 1024 can be factored as (8, 16, 8).
    FFT_SIZE = 8  # A smaller FFT size for demonstration purposes.
    FFT_FACTORS = (2, 2, 2)  # Factors for N=8 (2 * 2 * 2 = 8).
    # The 'D' parameter for data packing/unpacking in the kernel.
    # For N=8, N*2=16. ATOM_PACKING_DIM=2 is a valid divisor (16 % 2 == 0).
    ATOM_PACKING_DIM = 2

    # Data type for the real/imaginary components (e.g., float32 for complex64).
    PRECISION_DTYPE = torch.float32

    # --- Create Sample Input Data ---
    # Generate a random input tensor of complex64 numbers, placed on the CPU.
    # `torch.manual_seed(0)` ensures reproducibility of the random numbers.
    torch.manual_seed(0)
    input_data_complex = torch.randn(BATCH_SIZE, FFT_SIZE, dtype=torch.complex64, device='cpu')

    print("  Configuration:")
    print(f"  FFT Size (N): {FFT_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  FFT Factors (F0,F1,F2): {FFT_FACTORS}")
    print(f"  Atom Packing Dimension (D): {ATOM_PACKING_DIM}")
    print(f"Input data shape: {input_data_complex.shape}, dtype: {input_data_complex.dtype}")

    # Perform FFT using the custom cuTile kernel.
    output_fft_cutile = cutile_fft(
        x=input_data_complex,
        factors=FFT_FACTORS,
        atom_packing_dim=ATOM_PACKING_DIM,
        num_cpu_threads=args.num_cpu_threads,
    )
    print(
        f"""\ncuTile FFT Output shape: {output_fft_cutile.shape},
        dtype: {output_fft_cutile.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(output_fft_cutile, torch.fft.fft(input_data_complex, axis=-1))
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile FFT example execution complete ---")
