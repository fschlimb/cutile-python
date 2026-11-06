# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the built-in backends."""


def synchronize_stream(stream) -> None:
    """Wait for a framework stream; ``None`` and ``0`` stand for no stream."""
    if stream is None or (isinstance(stream, int) and stream == 0):
        return
    synchronize = getattr(stream, "synchronize", None)
    if synchronize is None:
        raise TypeError("stream must be None, 0, or provide synchronize()")
    synchronize()


def normalize_dims(value) -> tuple[int, int, int]:
    """Normalize one to three launch dimensions to a three-tuple."""

    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))


def check_options(backend: str, options, known) -> None:
    """Reject option keys the backend does not know, e.g. misspelled ones."""
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise ValueError(
            f"{backend} backend: unknown options {', '.join(map(repr, unknown))}; "
            f"expected any of {', '.join(sorted(known))}")


def bool_option(backend: str, options, key: str, default: bool) -> bool:
    """Read a boolean option, rejecting look-alikes such as ``"false"``."""

    value = options.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(
            f"{backend} backend: option {key!r} must be a bool, got {value!r}")
    return value
