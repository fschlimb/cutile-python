# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile import _backends as _backend
from cuda.tile._backends import ArrayArgument, _custom
from cuda.tile._backends._signature import build_signature, check_argument


@ct.kernel
def _kernel(x, scale: float, value: ct.Constant[int]):
    pass


class _Backend:
    name = "test"
    bytecode_version = "13.3"

    def __init__(self, identity="test:v1"):
        self.identity = identity
        self.calls = []

    def normalize_options(self, options):
        self.calls.append("normalize")
        return tuple(sorted(options.items()))

    def compiler_identity(self, options):
        return None if self.identity is None else f"{self.identity}:{options}"

    def compile(self, bytecode, signature, options):
        self.calls.append("compile")
        time.sleep(0.01)
        return b"binary"

    def load(self, binary, signature, options):
        self.calls.append("load")
        return binary, signature.symbol, options

    def launch(self, loaded, stream, grid, args):
        self.calls.append(("launch", loaded, grid))
        self.launched_args = args

    def synchronize(self, stream):
        self.calls.append("synchronize")


@pytest.fixture
def frontend(monkeypatch):
    """Replace the TileIR frontend and the tile context by recording fakes."""
    from cuda.tile import _compile

    calls = []

    def compile_tile(_function, signatures, *_args, **_kwargs):
        calls.append("tileir")
        return SimpleNamespace(kernel_signatures=list(signatures),
                               bytecode=b"tileir")

    monkeypatch.setattr(_compile, "compile_tile", compile_tile)
    config = SimpleNamespace(cache_dir=None, cache_size_limit=1 << 20)
    monkeypatch.setattr(_custom, "default_tile_context",
                        SimpleNamespace(config=config))
    return SimpleNamespace(calls=calls, config=config)


@pytest.fixture
def backend():
    backend = _Backend()
    _backend.set_backend(backend)
    yield backend
    _backend.set_backend(None)
    _kernel.__dict__.pop("_cutile_backend_cache", None)


def _args(scale=1.0, value=4):
    return np.zeros(16, dtype=np.float32), scale, value


def test_launch_compiles_and_loads_once(frontend, backend):
    x, _, _ = _args()
    ct.launch(None, (1,), _kernel, (x, 1.0, 4))
    ct.launch(None, (2,), _kernel, (x, 2.0, 4))

    launches = [call for call in backend.calls if call[0] == "launch"]
    assert frontend.calls == ["tileir"]
    assert backend.calls.count("compile") == 1
    assert backend.calls.count("load") == 1
    assert [grid for _, _, grid in launches] == [(1,), (2,)]
    assert launches[0][1] == (b"binary", build_signature(_kernel, _args()).symbol, ())


def test_launch_describes_each_array_once_and_drops_constants(
    frontend, backend, monkeypatch
):
    first = np.zeros(16, dtype=np.float32)
    second = np.ones(16, dtype=np.float32)
    described = []
    original_array_metadata = _custom.array_metadata

    def count_array_metadata(value):
        described.append(value)
        return original_array_metadata(value)

    monkeypatch.setattr(_custom, "array_metadata", count_array_metadata)

    ct.launch(None, (1,), _kernel, (first, 1.0, 4))
    first_args = backend.launched_args
    ct.launch(None, (1,), _kernel, (second, 2.0, 4))

    assert [value is first or value is second for value in described] == [True, False] * 2
    assert first_args == (
        ArrayArgument(first.ctypes.data, (16,), (1,), first.dtype, "cpu", 0), 1.0)
    assert backend.launched_args[0].pointer == second.ctypes.data
    assert backend.launched_args[1] == 2.0
    assert backend.calls.count("compile") == 1


def test_launch_compiled_passes_runtime_arguments(frontend, backend):
    x, _, _ = _args()
    compiled = _backend.compile_for_launch(_kernel, (x, 1.0, 4))

    _backend.launch_compiled(None, (3,), compiled, (x, 2.0, 4))

    assert backend.calls[-1][2] == (3,)
    assert backend.launched_args == (_custom.array_metadata(x), 2.0)
    with pytest.raises(TypeError, match="expects 3 arguments"):
        _backend.launch_compiled(None, (3,), compiled, (x, 2.0))


def test_caches_are_bounded(frontend, backend, monkeypatch):
    monkeypatch.setattr(_custom, "_CACHE_LIMIT", 2)
    x, _, _ = _args()

    for value in range(4):
        _backend.compile_for_launch(_kernel, (x, 1.0, value))

    cache = _kernel._cutile_backend_cache
    assert len(cache.launches) == len(cache.binaries) == 2
    _backend.compile_for_launch(_kernel, (x, 1.0, 3))
    assert backend.calls.count("compile") == 4
    _backend.compile_for_launch(_kernel, (x, 1.0, 0))
    assert backend.calls.count("compile") == 5


class _HostTensor:
    dtype = np.dtype("float32")

    def __init__(self, pointer=4096, shape=(16,), device="cpu"):
        self.pointer, self.shape, self.device = pointer, shape, device

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return (1,) * len(self.shape)


@pytest.fixture
def flat_backend(backend):
    def launch(loaded, stream, grid, args):
        signature = build_signature(_kernel, _args())
        check_argument("test", "cpu", signature.parameters[0], args[0])
        backend.calls.append("launch")

    def launch_flat(loaded, stream, grid, values):
        backend.calls.append(("launch_flat", tuple(values)))

    backend.launch = launch
    backend.launch_flat = launch_flat
    return backend


def _launch_kinds(backend):
    return [call if call == "launch" else call[0] for call in backend.calls
            if call == "launch" or call[0] == "launch_flat"]


def test_launch_flat_reuses_only_accepted_layouts(frontend, flat_backend):
    # 12288 has the same 4096-byte alignment as 4096; 4160 only has 64 bytes.
    for args in [(_HostTensor(), 1.0, 4), (_HostTensor(12288), 2.0, 4),
                 (_HostTensor(shape=(8,)), 1.0, 4), (_HostTensor(4160), 1.0, 4),
                 (_HostTensor(), 1.0, 5), (_HostTensor(), np.float64(1.0), 4)]:
        ct.launch(None, (1,), _kernel, args)

    assert _launch_kinds(flat_backend) == [
        "launch", "launch_flat", "launch", "launch", "launch", "launch"]
    assert ("launch_flat", (12288, 16, 1, 2.0)) in flat_backend.calls


def test_launch_flat_never_caches_rejected_layouts(frontend, flat_backend):
    for _ in range(2):
        with pytest.raises(ValueError, match="expected 'cpu'"):
            ct.launch(None, (1,), _kernel, (_HostTensor(device="cuda"), 1.0, 4))
    assert _launch_kinds(flat_backend) == []


def test_launch_flat_normalizes_integer_metadata(frontend, flat_backend):
    class Tensor(_HostTensor):
        def stride(self):
            return (1.0,)

    for tensor in (_HostTensor(), Tensor(shape=(16.0,)), Tensor(shape=(16.0,))):
        ct.launch(None, (1,), _kernel, (tensor, 1.0, 4))

    assert _launch_kinds(flat_backend) == ["launch", "launch_flat", "launch_flat"]
    for call in flat_backend.calls:
        if call[0] == "launch_flat":
            assert tuple(map(type, call[1])) == (int, int, int, float)


def test_launch_flat_distinguishes_missing_and_none_device(frontend, flat_backend):
    tensor = _HostTensor()
    del tensor.device
    ct.launch(None, (1,), _kernel, (tensor, 1.0, 4))
    ct.launch(None, (1,), _kernel, (tensor, 1.0, 4))

    for _ in range(2):
        with pytest.raises(ValueError, match="expected 'cpu'"):
            ct.launch(None, (1,), _kernel, (_HostTensor(device=None), 1.0, 4))
    assert _launch_kinds(flat_backend) == ["launch", "launch_flat"]


def test_launch_flat_falls_back_for_other_arrays(frontend, flat_backend):
    unhashable = SimpleNamespace(type="cpu", index=0)
    for args in [_args(), _args(), (_HostTensor(device=unhashable), 1.0, 4),
                 (_HostTensor(device=unhashable), 1.0, 4)]:
        ct.launch(None, (1,), _kernel, args)
    assert _launch_kinds(flat_backend) == ["launch"] * 4


def test_native_cache_unsupported_layout_does_not_insert(frontend, backend):
    class NativeCache:
        def launch(self, *args):
            return None

        def insert(self, *args):
            pytest.fail("unsupported layout must not be inserted")

    backend.make_launch_cache = lambda constants, limit: NativeCache()
    x, _, _ = _args()
    ct.launch(None, (1,), _kernel, (x, 1.0, 4))
    assert [call[0] for call in backend.calls if isinstance(call, tuple)] == ["launch"]


def test_native_caches_reuse_alternate_options(frontend, backend):
    class NativeCache:
        def __init__(self):
            self.lookups = 0

        def launch(self, *args):
            self.lookups += 1
            return False

        def insert(self, *args):
            pass

    caches = []

    def make_cache(constants, limit):
        cache = NativeCache()
        caches.append(cache)
        return cache

    backend.make_launch_cache = make_cache
    x, _, _ = _args()
    for option in (1, 2, 1):
        with ct.backend_options({"variant": option}):
            ct.launch(None, (1,), _kernel, (x, 1.0, 4))

    assert len(caches) == 2
    assert [cache.lookups for cache in caches] == [2, 1]


def test_native_cache_memo_is_a_consistent_snapshot(backend):
    parked = threading.Event()
    resume = threading.Event()
    launched = []

    class Options:
        def __init__(self, variant):
            self.variant = variant

        def __hash__(self):
            return hash(self.variant)

        def __eq__(self, other):
            if threading.current_thread() is not threading.main_thread():
                parked.set()
                assert resume.wait(5)
            return self.variant == other.variant

    class NativeCache:
        def __init__(self, variant):
            self.variant = variant

        def launch(self, args, stream, grid, options):
            launched.append((options["variant"], self.variant))
            return True

    backend.normalize_options = lambda options: Options(options["variant"])
    backend.make_launch_cache = lambda *args: NativeCache(
        _custom.current_options()["variant"])
    cache = _custom._kernel_cache(_kernel)
    first, second = NativeCache(1), NativeCache(2)
    key = (backend, Options(1))
    cache.native_launch = (key, first)
    cache.native_launches[key] = first
    cache.native_launches[(backend, Options(2))] = second

    def launch_first():
        with ct.backend_options({"variant": 1}):
            ct.launch(None, (0,), _kernel, _args())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(launch_first)
        try:
            assert parked.wait(5)
            with ct.backend_options({"variant": 2}):
                ct.launch(None, (0,), _kernel, _args())
        finally:
            resume.set()
        future.result(timeout=5)

    assert launched == [(2, 2), (1, 1)]


def test_runtime_scalar_values_do_not_rebuild_signatures(frontend, backend):
    built = []

    def builder(kernel, args):
        built.append(args[1])
        return build_signature(kernel, args)

    x, _, _ = _args()
    first = _backend.compile_for_launch(_kernel, (x, 1.0, 4), signature_builder=builder)
    second = _backend.compile_for_launch(_kernel, (x, 2.0, 4), signature_builder=builder)

    assert first is second
    assert built == [1.0]


def test_signature_builder_tracks_alignment_above_4096(frontend, backend):
    storage = np.zeros(32768, dtype=np.uint8)
    offset = -storage.ctypes.data % 8192
    aligned = storage[offset:offset + 16]
    less_aligned = storage[offset + 4096:offset + 4112]

    def builder(kernel, args):
        return build_signature(kernel, args, base_addr_divisible_by=8192)

    first = _backend.compile_for_launch(
        _kernel, (aligned, 1.0, 4), signature_builder=builder)
    second = _backend.compile_for_launch(
        _kernel, (less_aligned, 1.0, 4), signature_builder=builder)
    assert first is not second
    assert first.signature.parameters[0].base_addr_divisible_by == 8192
    assert second.signature.parameters[0].base_addr_divisible_by == 1
    assert _backend.compile_for_launch(
        _kernel, (aligned, 2.0, 4), signature_builder=builder) is first


def test_signature_builder_tracks_non_power_of_two_alignment(frontend, backend):
    class Tensor:
        shape, dtype, device = (16,), np.dtype("uint8"), "cpu"

        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def stride(self):
            return (1,)

    def builder(kernel, args):
        return build_signature(kernel, args, base_addr_divisible_by=3)

    first = _backend.compile_for_launch(
        _kernel, (Tensor(12288), 1.0, 4), signature_builder=builder)
    second = _backend.compile_for_launch(
        _kernel, (Tensor(20480), 1.0, 4), signature_builder=builder)
    assert first.signature.parameters[0].base_addr_divisible_by == 3
    assert second.signature.parameters[0].base_addr_divisible_by == 1


def test_constants_and_layouts_select_distinct_binaries(frontend, backend):
    x = np.zeros(32, dtype=np.float32)

    by_constant = _backend.compile_for_launch(_kernel, (x, 1.0, 8))
    by_layout = _backend.compile_for_launch(_kernel, (x[::2], 1.0, 4))
    base = _backend.compile_for_launch(_kernel, (x, 1.0, 4))

    assert len({id(by_constant), id(by_layout), id(base)}) == 3
    assert backend.calls.count("compile") == 3


@pytest.mark.parametrize("values", [
    (0.0, -0.0),
    (True, 1, 1.0),
    tuple(struct.unpack("=d", struct.pack("=Q", bits))[0]
          for bits in (0x7ff8000000000001, 0x7ff8000000000002)),
])
def test_constant_cache_preserves_types_and_float_bits(frontend, backend, values):
    x, _, _ = _args()
    compiled = [_backend.compile_for_launch(_kernel, (x, 1.0, value))
                for value in values]
    assert len({id(kernel) for kernel in compiled}) == len(values)
    for value, kernel in zip(values, compiled):
        assert _backend.compile_for_launch(_kernel, (x, 2.0, value)) is kernel
    assert backend.calls.count("compile") == len(values)


def test_same_signature_shares_the_loaded_binary(frontend, backend):
    x = np.zeros(64, dtype=np.float32)

    first = _backend.compile_for_launch(_kernel, (x[:16], 1.0, 4))
    second = _backend.compile_for_launch(_kernel, (x[16:32], 1.0, 4))

    assert first is second
    assert backend.calls.count("load") == 1


def test_options_are_normalized_once_and_select_binaries(frontend, backend):
    args = _args()
    with ct.backend_options({"level": 1}):
        first = _backend.compile_for_launch(_kernel, args)
        again = _backend.compile_for_launch(_kernel, args)
    with ct.backend_options({"level": 2}):
        second = _backend.compile_for_launch(_kernel, args)

    assert first is again
    assert first is not second
    assert first.options == (("level", 1),)
    assert backend.calls.count("normalize") == 2


def test_disk_cache_reuses_backend_binary(frontend, backend, tmp_path):
    frontend.config.cache_dir = str(tmp_path)

    _backend.compile_for_launch(_kernel, _args())
    _kernel.__dict__.pop("_cutile_backend_cache")
    _backend.compile_for_launch(_kernel, _args())

    assert frontend.calls == ["tileir", "tileir"]
    assert backend.calls.count("compile") == 1
    assert backend.calls.count("load") == 2


def test_unknown_identity_only_disables_the_disk_cache(frontend, backend, tmp_path):
    backend.identity = None
    frontend.config.cache_dir = str(tmp_path)

    first = _backend.compile_for_launch(_kernel, _args())
    again = _backend.compile_for_launch(_kernel, _args())
    _kernel.__dict__.pop("_cutile_backend_cache")
    _backend.compile_for_launch(_kernel, _args())

    assert first is again
    assert backend.calls.count("compile") == 2


def test_changed_identity_misses_the_disk_cache(frontend, backend, tmp_path):
    frontend.config.cache_dir = str(tmp_path)

    _backend.compile_for_launch(_kernel, _args())
    _kernel.__dict__.pop("_cutile_backend_cache")
    backend.identity = "test:v2"
    _backend.compile_for_launch(_kernel, _args())

    assert backend.calls.count("compile") == 2


def test_explicit_symbols_do_not_share_binaries(frontend, backend):
    def explicit(kernel, args):
        return build_signature(kernel, args).with_symbol("explicit")

    default = _backend.compile_for_launch(_kernel, _args())
    renamed = _backend.compile_for_launch(_kernel, _args(), signature_builder=explicit)

    assert renamed.symbol == "explicit"
    assert default.symbol != "explicit"


def test_concurrent_first_compile_runs_once(frontend, backend):
    args = _args()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: _backend.compile_for_launch(_kernel, args), range(4)))

    assert all(result is results[0] for result in results)
    assert backend.calls.count("compile") == 1


def test_benchmark_times_a_synchronized_launch(frontend, backend):
    elapsed = _backend.benchmark(None, (1,), _kernel, _args())

    assert elapsed >= 0
    assert backend.calls[-3:][0] == "synchronize"
    assert backend.calls[-2][0] == "launch"
    assert backend.calls[-1] == "synchronize"


def test_backend_protocol_lists_the_required_attributes():
    assert set(_custom._BACKEND_ATTRIBUTES) == {
        "name", "bytecode_version", "normalize_options", "compiler_identity",
        "compile", "load", "launch", "synchronize"}


def test_set_backend_rejects_incomplete_backends():
    with pytest.raises(TypeError, match="lacks .*load"):
        _backend.set_backend(SimpleNamespace(
            name="test", bytecode_version="13.3",
            normalize_options=None, compiler_identity=None, compile=None,
            launch=None, synchronize=None))
    assert _backend.get_backend() is None


@pytest.fixture
def cuda_launches(monkeypatch):
    launches = []
    monkeypatch.setattr(_custom, "_cext_launch",
                        lambda *args, **options: launches.append((args, options)))
    return launches


def test_launch_without_backend_uses_cuda(cuda_launches):
    ct.launch(None, (1,), _kernel, _args(), programmatic_dependent_launch=True)

    assert len(cuda_launches) == 1
    assert cuda_launches[0][1] == {"programmatic_dependent_launch": True}
    with pytest.raises(RuntimeError, match="no backend"):
        _backend.compile_for_launch(_kernel, _args())


class _CudaArray:
    __cuda_array_interface__ = {}


class _CudaTensor:
    shape, dtype = (16,), np.float32
    device = SimpleNamespace(type="cuda", index=0)

    def data_ptr(self):
        return 256

    def stride(self):
        return (1,)


@dataclass
class _Pair:
    left: object
    right: object


def test_cuda_tensor_uses_selected_backend(frontend, backend, cuda_launches):
    tensor = _CudaTensor()
    ct.launch(None, (1,), _kernel, (tensor, 1.0, 4))

    assert cuda_launches == []
    assert backend.launched_args == (_custom.array_metadata(tensor), 1.0)
    assert backend.calls.count("compile") == 1


def test_incompatible_device_raises_without_cuda_fallback(
        frontend, backend, cuda_launches, monkeypatch):
    def launch(loaded, stream, grid, args):
        signature = build_signature(_kernel, _args())
        check_argument("test", "cpu", signature.parameters[0], args[0])

    monkeypatch.setattr(backend, "launch", launch)
    with pytest.raises(ValueError, match="device 'cuda', expected 'cpu'"):
        ct.launch(None, (1,), _kernel, (_CudaTensor(), 1.0, 4))
    assert cuda_launches == []


def test_explicit_backend_switching(frontend, backend, cuda_launches):
    cpu_args = _args()
    cuda_args = (_CudaTensor(), 1.0, 4)
    ct.launch(None, (1,), _kernel, cpu_args)

    ct.set_backend(None)
    assert _backend.get_backend() is None
    ct.launch(None, (1,), _kernel, cuda_args, programmatic_dependent_launch=True)

    _backend.set_backend(backend)
    ct.launch(None, (2,), _kernel, cpu_args)

    assert cuda_launches == [
        ((None, (1,), _kernel, cuda_args), {"programmatic_dependent_launch": True})]
    assert backend.calls.count("compile") == 1
    assert [call[2] for call in backend.calls if call[0] == "launch"] == [(1,), (2,)]


@pytest.mark.parametrize("array", [_CudaArray(), _CudaTensor()])
@pytest.mark.parametrize("wrap", [
    lambda array: array,
    lambda array: (array,),
    lambda array: [array],
    lambda array: [(0, [array])],
    lambda array: _Pair(0, (array,)),
])
def test_cuda_arguments_do_not_enable_cuda_launch_options(
        frontend, backend, cuda_launches, array, wrap):
    args = (wrap(array), 1.0, 4)
    with pytest.raises(TypeError, match="only supported on CUDA"):
        ct.launch(None, (1,), _kernel, args, programmatic_dependent_launch=True)
    assert cuda_launches == []
    assert "compile" not in backend.calls


def test_cuda_tuning_timeout_is_allowed_after_clearing_backend(
        backend, monkeypatch):
    from cuda.tile.tune import _tune

    calls = []
    monkeypatch.setattr(_tune, "_benchmark", lambda *args: 1.0)
    monkeypatch.setattr(_tune, "benchmark_with_timeout",
                        lambda *args: calls.append(args))
    _backend.set_backend(None)
    result = ct.tune.exhaustive_search(
        [4], None, lambda _: (1,), _kernel,
        lambda value: (_CudaTensor(), 1.0, value),
        single_run_timeout_sec=1, quiet=True)
    assert result.best.config == 4
    assert len(calls) == 1


@pytest.mark.parametrize("array", [np.zeros(16, dtype=np.float32), _CudaTensor()])
def test_backend_tuning_timeout_is_rejected(backend, array):
    with pytest.raises(ValueError, match="not supported by custom backends"):
        ct.tune.exhaustive_search(
            [4], None, lambda _: (1,), _kernel,
            lambda value: (array, 1.0, value),
            single_run_timeout_sec=1, quiet=True)


def test_backends_reject_cuda_launch_options(frontend, backend):
    with pytest.raises(TypeError, match="only supported on CUDA"):
        ct.launch(None, (1,), _kernel, _args(), programmatic_dependent_launch=True)


@pytest.mark.parametrize("use_cuda", [False, True])
def test_tuning_benchmark_uses_selected_backend(backend, monkeypatch, use_cuda):
    from cuda.tile.tune import _tune

    calls = []
    monkeypatch.setattr(_backend, "benchmark",
                        lambda *args: (calls.append("backend") or 2.0))
    monkeypatch.setattr(_tune, "_benchmark",
                        lambda *args: (calls.append("cuda") or 1.0))
    if use_cuda:
        _backend.set_backend(None)

    elapsed = _tune._benchmark_launch(
        None, (1,), _kernel, (_CudaTensor(), 1.0, 4), False)

    assert calls == ["cuda" if use_cuda else "backend"]
    assert elapsed == (1.0 if use_cuda else 2.0)


def test_autotune_prepare_follows_explicit_backend_switch(
        frontend, backend, cuda_launches):
    tuned = ct.autotune(configs=[{"value": 4}], grid=(1,))(_kernel)
    args = (_CudaTensor(), 1.0)

    tuned.prepare(None, args)
    tuned.launch_prepared(None)
    assert backend.launched_args == (_custom.array_metadata(args[0]), 1.0)
    assert cuda_launches == []

    _backend.set_backend(None)
    tuned.prepare(None, args)
    tuned.launch_prepared(None)
    assert len(cuda_launches) == 1
    assert cuda_launches[0][0][3] == (*args, 4)


def test_autotune_compiles_each_config_once(frontend, backend):
    tuned = ct.autotune(configs=[{"value": 4}, {"value": 8}], grid=(1,))(_kernel)
    x, _, _ = _args()

    tuned(None, (x, 1.0))
    tuned(None, (x, 2.0))
    tuned.prepare(None, (x, 3.0))
    tuned.launch_prepared(None)

    assert backend.calls.count("compile") == 2
    assert tuned.best_config.meta["value"] in (4, 8)


def test_autotune_retunes_when_backend_changes(backend, monkeypatch):
    tuned = ct.autotune(configs=[{"value": 4}, {"value": 8}], grid=(1,))(_kernel)
    calls = []

    def tune(stream, args):
        calls.append(_backend.get_backend())
        return len(calls) - 1, None

    monkeypatch.setattr(tuned, "_tune", tune)
    args = _args()[:2]
    assert tuned._resolve(None, args) == 0
    _backend.set_backend(_Backend("other"))
    assert tuned._resolve(None, args) == 1
    _backend.set_backend(backend)
    assert tuned._resolve(None, args) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("callable_options", [False, True])
def test_autotune_retunes_when_launch_options_change(backend, monkeypatch, callable_options):
    options = (lambda meta: {"level": meta["scale"]}) if callable_options else None
    tuned = ct.autotune(configs=[{"value": 4}, {"value": 8}],
                        grid=(1,), options=options)(_kernel)
    calls = []

    def tune(stream, args):
        calls.append(args)
        return len(calls) - 1, None

    monkeypatch.setattr(tuned, "_tune", tune)
    x = _args()[0]
    for expected, value in ((0, 1), (1, 2), (0, 1)):
        with ct.backend_options({"num_cpu_threads": value}):
            assert tuned._resolve(None, (x, float(value))) == expected
    assert len(calls) == 2


def test_autotune_retunes_for_device_index(backend, monkeypatch):
    tuned = ct.autotune(configs=[{"value": 4}, {"value": 8}], grid=(1,))(_kernel)
    calls = []
    monkeypatch.setattr(tuned, "_tune",
                        lambda stream, args: (calls.append(args) or 0, None))
    tensor = _CudaTensor()
    for index in (0, 1, 0):
        tensor.device = SimpleNamespace(type="cuda", index=index)
        tuned._resolve(None, (tensor, 1.0))
    assert len(calls) == 2


def test_factory_autotune_retunes_when_environment_changes(backend, monkeypatch):
    tuned = ct.autotune(configs=[{"level": 1}, {"level": 2}])(
        lambda x, level=1: lambda: None)
    calls = []
    monkeypatch.setattr(tuned, "_tune",
                        lambda args, kwargs: (calls.append(args) or 0, None))
    x = _args()[0]
    for value in (1, 2, 1):
        with ct.backend_options({"num_cpu_threads": value}):
            tuned._resolve((x,), {})
    assert len(calls) == 2


def test_autotune_uses_overridden_options_not_ambient_options(backend, monkeypatch):
    tuned = ct.autotune(configs=[{"value": 4}, {"value": 8}],
                        grid=(1,), options={"level": 1})(_kernel)
    calls = []
    monkeypatch.setattr(tuned, "_tune",
                        lambda stream, args: (calls.append(args) or 0, None))
    args = _args()[:2]
    for value in (1, 2):
        with ct.backend_options({"level": value}):
            tuned._resolve(None, args)
    assert len(calls) == 1
