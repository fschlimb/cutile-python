# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

import cuda.tile as ct
from cuda.tile import _backend
from cuda.tile.compilation import CallingConvention, KernelSignature


@ct.kernel
def _kernel(value: ct.Constant[int]):
    pass


@pytest.fixture(autouse=True)
def clear_backend():
    _backend.clear_backend()
    yield
    _backend.clear_backend()


def _context(cache_dir=None):
    return SimpleNamespace(config=SimpleNamespace(
        cache_dir=cache_dir,
        cache_size_limit=1 << 20,
    ))


def _signature(value=1):
    return KernelSignature(
        [value], CallingConvention.cutile_python_v1())


def _fake_compile_tile(monkeypatch, calls):
    from cuda.tile import _compile

    def compile_tile(_function, signatures, *_args, **_kwargs):
        calls.append("tileir")
        return SimpleNamespace(
            kernel_signatures=list(signatures), bytecode=b"tileir")

    monkeypatch.setattr(_compile, "compile_tile", compile_tile)


def test_memory_cache_compiles_once(monkeypatch, capsys):
    """Reuse a custom backend binary from the per-kernel memory cache."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        return b"binary"

    _backend.set_backend(
        compile_fn=compile_tileir,
        compile_cache_key_fn=lambda: "test:v1",
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context()

    first = _kernel._compile(_signature(), context)
    second = _kernel._compile(_signature(), context)

    assert first == second
    assert calls == ["tileir", "backend"]
    assert capsys.readouterr().err == ""


def test_disk_cache_reuses_backend_binary(monkeypatch, tmp_path, capsys):
    """Reuse a backend binary from disk after clearing the memory cache."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        return b"binary"

    _backend.set_backend(
        compile_fn=compile_tileir,
        compile_cache_key_fn=lambda: "test:v1",
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context(str(tmp_path))

    _kernel._compile(_signature(), context)
    _kernel._custom_compile_cache.clear()
    result = _kernel._compile(_signature(), context)

    assert result[0] == b"binary"
    assert calls == ["tileir", "backend", "tileir"]
    assert capsys.readouterr().err == ""


def test_changed_identity_misses_memory_and_disk(monkeypatch, tmp_path):
    """Invalidate both caches when the backend compiler identity changes."""
    calls = []
    identity = ["test:v1"]
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        return identity[0].encode()

    _backend.set_backend(
        compile_fn=compile_tileir,
        compile_cache_key_fn=lambda: identity[0],
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context(str(tmp_path))

    assert _kernel._compile(_signature(), context)[0] == b"test:v1"
    identity[0] = "test:v2"
    assert _kernel._compile(_signature(), context)[0] == b"test:v2"

    assert calls == ["tileir", "backend", "tileir", "backend"]


def test_explicit_symbol_does_not_hide_signature_change(monkeypatch):
    """Keep distinct signatures separate despite a shared explicit symbol."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        return b"binary"

    _backend.set_backend(
        compile_fn=compile_tileir,
        compile_cache_key_fn=lambda: "test:v1",
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context()
    first = _signature(1).with_symbol("explicit")
    second = _signature(2).with_symbol("explicit")

    _kernel._compile(first, context)
    _kernel._compile(second, context)

    assert calls == ["tileir", "backend", "tileir", "backend"]


def test_backend_without_identity_warns_once(monkeypatch, capsys):
    """Disable caching and warn once when no cache-key callback is provided."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        return b"binary"

    _backend.set_backend(
        compile_fn=compile_tileir,
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context()

    _kernel._compile(_signature(), context)
    _kernel._compile(_signature(), context)

    assert calls == ["tileir", "backend", "tileir", "backend"]
    assert capsys.readouterr().err.count(
        "backend does not provide compile_cache_key()") == 1


def test_backend_without_compiler_identity_warns_once(monkeypatch, capsys):
    """Disable caching and warn once when compiler identity is unavailable."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)
    _backend.set_backend(
        compile_fn=lambda *_args, **_kwargs: b"binary",
        compile_cache_key_fn=lambda: None,
        sm_arch="3000",
        bytecode_version="13.3",
    )

    _kernel._compile(_signature(), _context())
    _kernel._compile(_signature(), _context())

    assert capsys.readouterr().err.count(
        "backend could not determine its compiler identity") == 1


def test_invalid_cache_identity_is_rejected():
    """Reject compiler identities that are neither strings nor bytes."""
    _backend.set_backend(
        compile_fn=lambda *_args, **_kwargs: b"binary",
        compile_cache_key_fn=lambda: object(),
        sm_arch="3000",
        bytecode_version="13.3",
    )

    with pytest.raises(TypeError, match="compile_cache_key"):
        _kernel._compile(_signature(), _context())


def test_module_compiler_override_does_not_inherit_cache_key(monkeypatch):
    """Avoid pairing an overridden compiler with a module's cache identity."""
    module = SimpleNamespace(
        compile_tileir=lambda *_args, **_kwargs: b"module",
        compile_cache_key=lambda: "module:v1",
    )
    override = lambda *_args, **_kwargs: b"override"
    monkeypatch.setattr(
        _backend, "_import_backend_module", lambda name: module)

    _backend.set_backend("module", compile_fn=override)

    assert _backend.get_compile_fn() is override
    assert _backend.get_compile_cache_key_fn() is None


def test_concurrent_first_compile_runs_once(monkeypatch):
    """Compile once when concurrent callers miss the memory cache together."""
    calls = []
    _fake_compile_tile(monkeypatch, calls)

    def compile_tileir(*_args, **_kwargs):
        calls.append("backend")
        time.sleep(0.02)
        return b"binary"

    _backend.set_backend(
        compile_fn=compile_tileir,
        compile_cache_key_fn=lambda: "test:concurrent",
        sm_arch="3000",
        bytecode_version="13.3",
    )
    context = _context()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: _kernel._compile(_signature(), context), range(4)))

    assert all(result[0] == b"binary" for result in results)
    assert calls == ["tileir", "backend"]
