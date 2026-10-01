# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import time
from types import SimpleNamespace

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile import _backend
from cuda.tile._backend import _custom
from cuda.tile._backend._signature import build_signature


@ct.kernel
def _kernel(x, scale: float, value: ct.Constant[int]):
    pass


class _Backend:
    sm_arch = "3000"
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
    _backend.clear_backend()
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


def test_launch_shares_fresh_argument_metadata_with_backend(
    frontend, backend, monkeypatch
):
    first = np.zeros(16, dtype=np.float32)
    second = np.ones(16, dtype=np.float32)
    metadata_calls = []
    launches = []
    original_array_metadata = _custom.array_metadata

    def count_array_metadata(value):
        if isinstance(value, np.ndarray):
            metadata_calls.append(value)
        return original_array_metadata(value)

    def launch_with_metadata(loaded, stream, grid, args, argument_metadata):
        launches.append((args, argument_metadata))
        backend.launch(loaded, stream, grid, args)

    monkeypatch.setattr(_custom, "array_metadata", count_array_metadata)
    monkeypatch.setattr(
        backend, "_launch_with_metadata", launch_with_metadata, raising=False
    )

    ct.launch(None, (1,), _kernel, (first, 1.0, 4))
    ct.launch(None, (1,), _kernel, (second, 2.0, 4))

    assert len(metadata_calls) == 2
    assert metadata_calls[0] is first
    assert metadata_calls[1] is second
    assert launches[0][0][0] is first
    assert launches[1][0][0] is second
    assert launches[0][1][0][0] == first.ctypes.data
    assert launches[1][1][0][0] == second.ctypes.data
    assert backend.calls.count("compile") == 1


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


def test_constants_and_layouts_select_distinct_binaries(frontend, backend):
    x = np.zeros(32, dtype=np.float32)

    by_constant = _backend.compile_for_launch(_kernel, (x, 1.0, 8))
    by_layout = _backend.compile_for_launch(_kernel, (x[::2], 1.0, 4))
    base = _backend.compile_for_launch(_kernel, (x, 1.0, 4))

    assert len({id(by_constant), id(by_layout), id(base)}) == 3
    assert backend.calls.count("compile") == 3


def test_same_signature_shares_the_loaded_binary(frontend, backend):
    x = np.zeros(64, dtype=np.float32)

    first = _backend.compile_for_launch(_kernel, (x[:16], 1.0, 4))
    second = _backend.compile_for_launch(_kernel, (x[16:32], 1.0, 4))

    assert first is second
    assert backend.calls.count("load") == 1


def test_options_are_normalized_once_and_select_binaries(frontend, backend):
    args = _args()
    with _backend.compile_options({"level": 1}):
        first = _backend.compile_for_launch(_kernel, args)
        again = _backend.compile_for_launch(_kernel, args)
    with _backend.compile_options({"level": 2}):
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


def test_set_backend_rejects_incomplete_backends():
    with pytest.raises(TypeError, match="lacks .*load"):
        _backend.set_backend(SimpleNamespace(
            sm_arch="3000", bytecode_version="13.3",
            normalize_options=None, compiler_identity=None, compile=None,
            launch=None, synchronize=None))
    assert _backend.get_backend() is None


def test_launch_without_backend_uses_cuda(monkeypatch):
    launches = []
    monkeypatch.setattr(_custom, "_cext_launch", lambda *args: launches.append(args))

    ct.launch(None, (1,), _kernel, _args())

    assert len(launches) == 1
    with pytest.raises(RuntimeError, match="no backend"):
        _backend.compile_for_launch(_kernel, _args())
