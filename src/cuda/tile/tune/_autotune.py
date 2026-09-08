# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Triton-style ``@autotune`` for cuTile kernels and multi-kernel factories."""
from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

from cuda.tile import _backend, _execution
from cuda.tile._backend._custom import compile_for_launch
from cuda.tile._backend._signature import array_metadata
from cuda.tile.tune._tune import exhaustive_search

@dataclass(frozen=True, init=False)
class Config:
    """A set of compile-time values and compiler hints for autotuning."""

    meta: Mapping[str, Any]
    hints: Mapping[str, Any]

    def __init__(self, meta: Mapping[str, Any], /, **hints: Any):
        if not isinstance(meta, Mapping):
            raise TypeError("Config meta must be a mapping")
        values = dict(meta)
        if any(not isinstance(name, str) for name in values):
            raise TypeError("Config meta names must be strings")
        object.__setattr__(self, "meta", MappingProxyType(values))
        object.__setattr__(self, "hints", MappingProxyType(dict(hints)))

    def __repr__(self) -> str:
        hints = "".join(f", {name}={value!r}"
                        for name, value in self.hints.items())
        return f"Config({dict(self.meta)!r}{hints})"


def _normalize_configs(configs, valid_names, what):
    """Validate ``configs`` and return them with the set of tuned names."""

    normalized = tuple(config if isinstance(config, Config) else Config(config)
                       for config in configs)
    if not normalized:
        raise ValueError("autotune configs must not be empty")
    tuned = set(normalized[0].meta)
    for config in normalized:
        unknown = set(config.meta) - valid_names
        if unknown:
            raise ValueError(
                f"Config contains unknown {what}: {sorted(unknown)}")
        if set(config.meta) != tuned:
            raise ValueError(f"All autotune configs must use the same {what}")
    return normalized, tuned


def _key_names(key, valid_names, tuned):
    names = tuple(key)
    if len(set(names)) != len(names):
        raise ValueError("autotune key names must be unique")
    unknown = set(names) - valid_names
    if unknown:
        raise ValueError(
            f"autotune key contains unknown parameters: {sorted(unknown)}")
    overlap = set(names) & tuned
    if overlap:
        raise ValueError(
            f"autotune key cannot contain autotuned parameters: {sorted(overlap)}")
    return set(names)


def _array_key(value):
    """Layout fingerprint of an array value, or ``None`` if not an array."""

    try:
        _pointer, shape, strides, dtype = array_metadata(value)
    except TypeError:
        return None
    return str(getattr(value, "device", "")), dtype, shape, strides


def _config_key(named_values, extra_names):
    """Key deciding which tuning result applies to ``named_values``.

    Arrays contribute their layout only, so a tuning result is shared between
    distinct buffers of the same shape. Other values contribute only when
    named, either explicitly via ``key`` or because they are compile-time
    constants.
    """

    key = []
    for name, value in named_values:
        array = _array_key(value)
        if array is not None:
            key.append((name, array))
        elif name in extra_names:
            key.append((name, type(value), value))
    return tuple(key)


def _make_injector(positions, values):
    """Build a callable inserting ``values`` at ``positions`` (ascending)."""

    def inject(args):
        full = list(args)
        for position, value in zip(positions, values):
            full.insert(position, value)
        return tuple(full)

    return inject


class _Autotuned:
    """Config cache shared by the two autotuning flavors."""

    _configs: tuple[Config, ...]

    def __init__(self):
        self._cache: dict[Any, int] = {}
        self._best: int | None = None
        self._prepared = None

    @property
    def best_config(self) -> Config | None:
        """The config selected by the most recent call."""

        return None if self._best is None else self._configs[self._best]

    def _index_by_id(self):
        return {id(config): index for index, config in enumerate(self._configs)}

    def _cached_index(self, key):
        index = self._cache.get(key)
        if index is not None:
            self._best = index
        return index

    def _store_index(self, key, index):
        self._cache[key] = index
        self._best = index


class AutotunedKernel(_Autotuned):
    """A lazily tuned wrapper around a :class:`cuda.tile.kernel`.

    Configs supply a subset of the kernel's ``ct.Constant`` parameters; those
    parameters are omitted from the argument tuple passed at launch.
    """

    def __init__(self, kernel, configs: Sequence[Config | Mapping[str, Any]],
                 *, key: Sequence[str] = (), grid, options=None,
                 quiet: bool = True):
        super().__init__()
        annotated = getattr(kernel, "_annotated_function", None)
        if annotated is None:
            raise TypeError("autotune must wrap a cuda.tile.kernel")
        if grid is None:
            raise TypeError("autotune requires a grid callable or tuple")
        if options is not None and not callable(options):
            if not isinstance(options, Mapping):
                raise TypeError("autotune options must be a mapping or callable")
            options = dict(options)

        names = tuple(annotated.pysig.parameters)
        constants = {name for name, is_constant
                     in zip(names, annotated.constant_parameter_mask)
                     if is_constant}
        self._configs, tuned = _normalize_configs(configs, constants,
                                                  "constant parameters")
        # Untuned constants stay caller arguments but are still compile-time
        # values, so they have to take part in config selection.
        self._extra_names = (_key_names(key, set(names), tuned)
                             | (constants - tuned))
        positions = tuple(index for index, name in enumerate(names)
                          if name in tuned)

        self._kernel = kernel
        self._names = names
        self._arg_names = tuple(name for name in names if name not in tuned)
        self._kernels = tuple(
            kernel.replace_hints(**config.hints) if config.hints else kernel
            for config in self._configs)
        self._injectors = tuple(
            _make_injector(positions,
                           tuple(config.meta[names[i]] for i in positions))
            for config in self._configs)
        self._grid = grid
        self._options = options
        self._quiet = quiet

    @property
    def kernel(self):
        return self._kernel

    def _plan_for(self, index, args):
        grid = self._grid
        options = self._options
        if callable(grid) or callable(options):
            full_args = self._injectors[index](args)
            meta = dict(zip(self._names, full_args))
            if callable(grid):
                grid = grid(meta)
            if callable(options):
                options = options(meta)
                if not isinstance(options, Mapping):
                    raise TypeError(
                        "autotune options callable must return a mapping")
                options = dict(options)
        return ((grid,) if isinstance(grid, int) else tuple(grid), options)

    @staticmethod
    def _options_context_factory(options):
        if options is None:
            return None
        factory = _backend.get_compile_options_fn()
        if factory is None:
            raise RuntimeError(
                "active backend does not provide compile_options()")
        return partial(factory, options)

    def _tune(self, stream, args):
        index_by_id = self._index_by_id()
        plans = {}

        def plan_for(config):
            index = index_by_id[id(config)]
            plan = plans.get(index)
            if plan is None:
                plan = self._plan_for(index, args)
                plans[index] = plan
            return plan

        search_kwargs = {}
        if self._options is not None:
            search_kwargs["context_fn"] = lambda config: (
                self._options_context_factory(plan_for(config)[1])())
        result = exhaustive_search(
            self._configs,
            stream,
            grid_fn=lambda config: plan_for(config)[0],
            kernel=self._kernel,
            args_fn=lambda config: self._injectors[index_by_id[id(config)]](args),
            hints_fn=lambda config: config.hints,
            quiet=self._quiet,
            **search_kwargs,
        )
        return index_by_id[id(result.best.config)]

    def _resolve(self, stream, args):
        if len(args) != len(self._arg_names):
            raise TypeError(
                f"autotuned kernel expects {len(self._arg_names)} arguments, "
                f"got {len(args)}")
        key = _config_key(zip(self._arg_names, args), self._extra_names)
        index = self._cached_index(key)
        if index is None:
            index = self._tune(stream, args)
            self._store_index(key, index)
        return index

    def prepare(self, stream, kernel_args, /) -> Config:
        """Tune and compile for ``kernel_args`` ahead of timing.

        Use :meth:`launch_prepared` to benchmark without config lookup,
        constant injection or grid computation in the measured call.
        """

        args = tuple(kernel_args)
        index = self._resolve(stream, args)
        grid, options = self._plan_for(index, args)
        full_args = self._injectors[index](args)
        kernel = self._kernels[index]
        launch_compiled_fn = _backend.get_launch_compiled_fn()
        context_factory = self._options_context_factory(options)
        if launch_compiled_fn is None:
            target = kernel
        else:
            if context_factory is None:
                target = compile_for_launch(kernel, full_args)
            else:
                with context_factory():
                    target = compile_for_launch(kernel, full_args)
        self._prepared = (launch_compiled_fn or _execution.launch, grid, target,
                          full_args, context_factory)
        return self._configs[index]

    def launch_prepared(self, stream, /):
        """Launch what the most recent :meth:`prepare` call built.

        This path deliberately skips all lookups, so it stays bound to the
        arguments that were passed to :meth:`prepare`.
        """

        if self._prepared is None:
            raise RuntimeError("autotuned kernel must be prepared before launch")
        launch_fn, grid, target, args, context_factory = self._prepared
        if context_factory is None:
            return launch_fn(stream, grid, target, args)
        with context_factory():
            return launch_fn(stream, grid, target, args)

    def __call__(self, stream, kernel_args, /):
        """Launch with the best known config for ``kernel_args``."""

        args = tuple(kernel_args)
        index = self._resolve(stream, args)
        grid, options = self._plan_for(index, args)
        if options is None:
            return _execution.launch(stream, grid, self._kernels[index],
                                     self._injectors[index](args))
        with self._options_context_factory(options)():
            return _execution.launch(stream, grid, self._kernels[index],
                                     self._injectors[index](args))


class AutotunedFunction(_Autotuned):
    """Autotune a host factory that returns a prepared callable.

    Each config overrides named factory parameters; the returned callable may
    launch any number of kernels.
    """

    def __init__(self, function, configs: Sequence[Config | Mapping[str, Any]],
                 *, key: Sequence[str] = (), quiet: bool = True):
        super().__init__()
        self._function = function
        self._signature = inspect.signature(function)
        names = set(self._signature.parameters)
        self._configs, tuned = _normalize_configs(configs, names,
                                                  "function parameters")
        if any(config.hints for config in self._configs):
            raise ValueError(
                "Compiler hints are only supported when autotuning a ct.kernel")
        self._extra_names = _key_names(key, names, tuned)
        self._quiet = quiet

    def _bind(self, args, kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def _build(self, index, args, kwargs):
        bound = self._bind(args, kwargs)
        bound.arguments.update(self._configs[index].meta)
        prepared = self._function(*bound.args, **bound.kwargs)
        if not callable(prepared):
            raise TypeError(
                "autotuned function must return a callable prepared operation")
        return prepared

    def _tune(self, args, kwargs):
        timer = _backend.get_benchmark_callable_fn()
        if timer is None:
            raise RuntimeError(
                "active backend does not provide benchmark_callable()")
        index_by_id = self._index_by_id()
        built: dict[int, Any] = {}

        def args_fn(config):
            index = index_by_id[id(config)]
            if index not in built:
                built[index] = self._build(index, args, kwargs)
            return (built[index],)

        result = exhaustive_search(
            self._configs,
            None,
            grid_fn=lambda _config: (1,),
            kernel=None,
            args_fn=args_fn,
            hints_fn=None,
            quiet=self._quiet,
            benchmark_fn=lambda stream, _grid, _kernel, bench_args: timer(
                stream, bench_args[0], ()),
        )
        index = index_by_id[id(result.best.config)]
        return index, built[index]

    def _resolve(self, args, kwargs):
        """Return the config index and the operation built while tuning, if any."""

        key = _config_key(self._bind(args, kwargs).arguments.items(),
                          self._extra_names)
        index = self._cached_index(key)
        if index is not None:
            return index, None
        index, prepared = self._tune(args, kwargs)
        self._store_index(key, index)
        return index, prepared

    def prepare(self, *args, **kwargs) -> Config:
        """Build the prepared multi-kernel operation ahead of timing."""

        index, prepared = self._resolve(args, kwargs)
        self._prepared = (prepared if prepared is not None
                          else self._build(index, args, kwargs))
        return self._configs[index]

    def launch_prepared(self):
        """Run the operation built by the most recent :meth:`prepare` call."""

        if self._prepared is None:
            raise RuntimeError(
                "autotuned function must be prepared before launch")
        return self._prepared()

    def __call__(self, *args, **kwargs):
        index, prepared = self._resolve(args, kwargs)
        if prepared is None:
            prepared = self._build(index, args, kwargs)
        return prepared()


def autotune(*,
             configs: Sequence[Config | Mapping[str, Any]],
             key: Sequence[str] = (),
             grid=None,
             options=None,
             quiet: bool = True):
    """Autotune a cuTile kernel or a factory for one or more kernel launches.

    ``configs`` is the search space. For a kernel, each config may provide a
    subset of its ``ct.Constant`` parameters and compiler hints. Configured
    constants are inserted into the kernel arguments automatically. For a
    regular Python function, config entries override parameters by name and
    the function must return a callable prepared operation. This supports
    factories that prepare and capture several kernel launches.

    Kernel mode requires ``grid``. It can be a fixed integer or tuple, or a
    callable with the form ``grid(meta)``. ``meta`` is a dictionary containing
    every kernel parameter, including runtime arguments and configured
    constants. This lets the grid depend on both input shapes and the config::

        @ct.autotune(
            configs=[
                ct.tune.Config({"TILE": 32}),
                ct.tune.Config({"TILE": 64}),
            ],
            grid=lambda meta: (
                ct.cdiv(meta["x"].shape[0], meta["TILE"]),
            ),
            options=lambda meta: {
                "assume_in_bounds": meta["x"].shape[0] % meta["TILE"] == 0,
            },
        )
        @ct.kernel
        def add_one(x, y, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            value = ct.load(x, index=(bid,), shape=(TILE,))
            ct.store(y, index=(bid,), tile=value + 1)

        add_one.prepare(stream, (x, y))
        add_one.launch_prepared(stream)

    ``options`` accepts a fixed backend-options mapping or an
    ``options(meta)`` callback. Like ``grid(meta)``, the callback receives the
    complete parameter dictionary after configured constants are inserted and
    must return an options mapping. Use it when a backend option depends on a
    selected config, for example to enable CPU bounds assumptions only when a
    tuned block size divides the input length::

        @ct.autotune(
            configs=[
                ct.tune.Config({"BLOCK_SIZE": 256}),
                ct.tune.Config({"BLOCK_SIZE": 512}),
            ],
            grid=lambda meta: (
                ct.cdiv(meta["n_elements"], meta["BLOCK_SIZE"]),
            ),
            options=lambda meta: {
                "assume_in_bounds": (
                    meta["n_elements"] % meta["BLOCK_SIZE"] == 0
                ),
            },
        )
        @ct.kernel
        def leaky_relu(x, output, n_elements: ct.Constant[int],
                       BLOCK_SIZE: ct.Constant[int]):
            ...

    The resolved options are active while each candidate is tuned, and while
    the selected kernel is compiled and launched. This requires a backend that
    provides ``compile_options``.

    Factory mode omits ``grid`` because the returned prepared callable owns
    the grids for its launches. The factory example below tunes a parameter
    that changes a multi-kernel operation and keeps tuning, preparation, and
    config lookup outside the measured region::

        @ct.autotune(
            configs=[
                ct.tune.Config({"blocking_factor_k": 1}),
                ct.tune.Config({"blocking_factor_k": 2}),
            ],
        )
        def prepare_matmul(a, b, *, blocking_factor_k=1):
            return prepare_sfc_matmul(
                a, b, blocking_factor_k=blocking_factor_k)

        prepare_matmul.prepare(a, b)
        result = prepare_matmul.launch_prepared()

    On the first call for a new input layout or key, every config is prepared
    and benchmarked. Later calls reuse the selected config. Array arguments are
    keyed by device, dtype, shape, and strides; use ``key`` for additional
    scalar parameters that affect the generated work. Use ``prepare`` followed
    by ``launch_prepared`` when measuring kernel time: the latter performs no
    tuning, compilation, grid calculation, or argument preparation. The
    ``stream`` argument is required in kernel mode and is omitted in factory
    mode.

    Args:
        configs: Non-empty sequence of :class:`Config` objects or metadata
            mappings. All configs in one decorator must name the same
            parameters.
        key: Optional names of scalar runtime parameters that should select a
            separate tuning result. Array layouts are included automatically.
        grid: Fixed launch geometry or ``grid(meta)`` callable for kernel mode.
            It is invalid for factory mode.
        options: Fixed backend compile options mapping or ``options(meta)``
            callback for kernel mode. The callback receives configured
            constants and runtime parameters and must return a mapping. The
            selected options are active while tuning, compiling, and
            launching. It is invalid for factory mode.
        quiet: If true, suppress the tuning summary and progress output.

    Returns:
        A callable wrapper with ``prepare`` and ``launch_prepared`` methods,
        plus ``best_config``. Kernel wrappers are called as
        ``wrapper(stream, kernel_args)``; factory wrappers are called with the
        factory's normal arguments.

    Raises:
        ValueError: If configs or key names are invalid.
        TypeError: If the decorated object or grid form is invalid.
    """

    def decorate(kernel_or_function):
        if getattr(kernel_or_function, "_annotated_function", None) is not None:
            return AutotunedKernel(kernel_or_function, configs, key=key,
                                   grid=grid, options=options, quiet=quiet)
        if grid is not None:
            raise TypeError("grid is only valid when autotuning a ct.kernel")
        if options is not None:
            raise TypeError("options is only valid when autotuning a ct.kernel")
        return AutotunedFunction(kernel_or_function, configs, key=key,
                                 quiet=quiet)

    return decorate


__all__ = ["Config", "AutotunedKernel", "AutotunedFunction", "autotune"]
