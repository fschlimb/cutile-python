# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Vendored from the Lighthouse project (lighthouse/pipeline/xegpu/xeas.py),
# with the transform-schedule driver replaced by a direct pass-manager run.
"""xeas - Xe assembler.

Compiles an "outlined" MLIR payload down to a serialized GPU kernel binary blob.

The input is an outlined kernel: a ``gpu.module`` containing a single
``gpu.func`` (the vectorized kernel body), and nothing else -- no host
``func.func`` and no ``gpu.launch_func`` launcher. This is the IR produced by
the XeGPU pipeline's ``outlined`` stage.

xeas runs the upstream ``gpu-lower-to-xevm-pipeline`` on that IR to produce a
``gpu.binary`` op, then returns the embedded device object -- the exact byte
string the Level Zero runtime loads at run time (``mgpuModuleLoad``). The blob
can be launched with
:func:`cuda.tile._level_zero.launch_level_zero_module_kernel`
without producing a host-side launcher or shared library.

The high-level entry point is :func:`xeas`:

    from cuda.tile._backend.xeas import xeas

    blob = xeas(outlined_mlir)
"""

from mlir import ir
from mlir.dialects import gpu
from mlir.passmanager import PassManager


def _select_gpu_object(objects: ir.ArrayAttr, large_register_file: bool):
    """Pick the ``#gpu.object`` matching the large-register-file setting.

    With a single object the choice is unambiguous; otherwise the object whose
    target carries (or omits) the ``-ze-opt-large-register-file`` IGC option is
    selected to match ``large_register_file``, falling back to the first object.
    """
    if len(objects) == 1:
        return gpu.ObjectAttr(objects[0])

    def uses_lrf(obj) -> bool:
        return "large-register-file" in str(gpu.ObjectAttr(obj).target)

    for obj in objects:
        if uses_lrf(obj) == large_register_file:
            return gpu.ObjectAttr(obj)
    return gpu.ObjectAttr(objects[0])


def extract_gpu_binary(module: ir.Module, *, large_register_file: bool = True) -> bytes:
    """Extract the serialized GPU kernel binary from a lowered module.

    The XeGPU-to-binary pipeline embeds the device kernel in a ``gpu.binary`` op
    as one or more ``#gpu.object`` attributes (one per target attached to the
    ``gpu.module``). This returns the raw object bytes -- the blob the Level Zero
    runtime loads at run time via ``mgpuModuleLoad``.

    When several objects are present (e.g. a plain target and one built with the
    large register file IGC option), the object matching ``large_register_file``
    is returned.
    """
    binaries = [
        op for op in module.body.operations if op.operation.name == "gpu.binary"
    ]
    if not binaries:
        raise ValueError("no 'gpu.binary' op found in the lowered module")
    if len(binaries) > 1:
        raise ValueError(f"expected a single 'gpu.binary' op, found {len(binaries)}")
    objects = ir.ArrayAttr(binaries[0].attributes["objects"])
    if len(objects) == 0:
        raise ValueError("'gpu.binary' op has no embedded objects")
    return _select_gpu_object(objects, large_register_file).object


def _xevm_pipeline(
    xegpu_op_level: str,
    large_register_file: bool,
    enable_vector_to_xegpu: bool = True,
) -> str:
    """Build the textual pass pipeline lowering XeGPU IR to a device binary."""
    options = [
        f"xegpu-op-level={xegpu_op_level}",
        f"enable-vector-to-xegpu={'true' if enable_vector_to_xegpu else 'false'}",
    ]
    if large_register_file:
        options.append("igc-cmd-options=-ze-opt-large-register-file")
    return f"builtin.module(gpu-lower-to-xevm-pipeline{{{' '.join(options)}}})"


def lower_payload(
    source: str,
    *,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
) -> ir.Module:
    """Lower an outlined MLIR kernel to a module containing its ``gpu.binary``.

    Runs ``gpu-lower-to-xevm-pipeline`` on the ``outlined`` stage IR (a
    ``gpu.module`` with a single ``gpu.func``). This is the context-bound part
    of :func:`xeas` and must run inside an active MLIR context.

    Args:
        source: MLIR text at the ``outlined`` stage.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.

    Returns:
        The lowered module containing the embedded ``gpu.binary`` kernel.
    """
    with ir.Location.unknown():
        module = ir.Module.parse(source)
        pipeline = _xevm_pipeline(xegpu_op_level, large_register_file)
        PassManager.parse(pipeline, module.context).run(module.operation)
        return module


def xeas(
    source: str,
    *,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
) -> bytes:
    """Compile an outlined MLIR kernel into a GPU kernel binary blob.

    Lowers ``source`` with :func:`lower_payload` and extracts the embedded
    device kernel with :func:`extract_gpu_binary`, returning the blob bytes --
    the exact byte string the Level Zero runtime loads at run time
    (``mgpuModuleLoad``).
    :func:`cuda.tile._level_zero.launch_level_zero_module_kernel`
    launches the kernel directly from it.

    Args:
        source: MLIR text at the ``outlined`` stage.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.

    Returns:
        The serialized GPU kernel binary as bytes.
    """
    module = lower_payload(
        source,
        xegpu_op_level=xegpu_op_level,
        large_register_file=large_register_file,
    )
    return extract_gpu_binary(module, large_register_file=large_register_file)


__all__ = ["extract_gpu_binary", "lower_payload", "xeas"]
