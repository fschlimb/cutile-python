# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import runpy

import pytest
import setuptools


@pytest.fixture
def link_extension(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup = Path(__file__).resolve().parents[2] / "setup.py"
    return runpy.run_path(str(setup))["_link_editable_extension"]


def test_existing_editable_link_is_reused(link_extension, tmp_path):
    source = tmp_path / "source.so"
    source.write_bytes(b"new")
    destination = tmp_path / "extension.so"
    destination.symlink_to(source)

    link_extension(str(source), str(destination))

    assert destination.is_symlink()
    assert os.readlink(destination) == str(source)
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("old_destination", ["other-link", "broken-link", "regular-file"])
def test_editable_link_replaces_stale_destination(link_extension, tmp_path, old_destination):
    source = tmp_path / "source.so"
    source.write_bytes(b"new")
    destination = tmp_path / "extension.so"
    if old_destination == "regular-file":
        destination.write_bytes(b"old")
    else:
        old_source = tmp_path / "old.so"
        if old_destination == "other-link":
            old_source.write_bytes(b"old")
        destination.symlink_to(old_source)

    link_extension(str(source), str(destination))

    assert destination.is_symlink()
    assert os.readlink(destination) == str(source)
    assert destination.read_bytes() == b"new"


def test_missing_source_preserves_editable_link(link_extension, tmp_path):
    destination = tmp_path / "extension.so"
    destination.write_bytes(b"old")

    with pytest.raises(FileNotFoundError):
        link_extension(str(tmp_path / "missing.so"), str(destination))

    assert destination.read_bytes() == b"old"
