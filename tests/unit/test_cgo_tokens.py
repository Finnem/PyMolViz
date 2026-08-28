"""Unit tests for CGO token resolution."""

from __future__ import annotations

import pytest

from pymolviz.util.cgo import resolve_cgo_tokens


def test_numeric_string_becomes_float():
    from pymol import cgo

    out = resolve_cgo_tokens(["COLOR", "0.86", "1.0", "0.0"])
    assert out[0] == cgo.COLOR
    assert out[1] == pytest.approx(0.86)
    assert out[2] == pytest.approx(1.0)


def test_known_token_string():
    from pymol import cgo

    out = resolve_cgo_tokens(["BEGIN", "TRIANGLES"])
    assert out[0] == cgo.BEGIN
    assert out[1] == cgo.TRIANGLES


def test_float_passthrough():
    out = resolve_cgo_tokens([0.5, 1.25])
    assert out == [pytest.approx(0.5), pytest.approx(1.25)]


def test_unknown_token_raises():
    with pytest.raises(KeyError):
        resolve_cgo_tokens(["NOT_A_REAL_CGO_TOKEN"])
