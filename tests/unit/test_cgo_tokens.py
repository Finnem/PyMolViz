"""Unit tests for CGO token resolution."""

from __future__ import annotations

import pytest

from pymolviz.util.cgo import offset_cgo_vertices, resolve_cgo_tokens


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


def test_offset_cgo_vertices_shifts_vertex_and_sphere():
    content = [
        "BEGIN", "TRIANGLES",
        "VERTEX", 1.0, 2.0, 3.0,
        "END",
        "COLOR", 1.0, 0.0, 0.0,
        "SPHERE", 4.0, 5.0, 6.0, 0.5,
    ]
    offset_cgo_vertices(content, (1.0, -2.0, 0.5))
    assert content[3:6] == [pytest.approx(2.0), pytest.approx(0.0), pytest.approx(3.5)]
    assert content[12:16] == [
        pytest.approx(5.0), pytest.approx(3.0), pytest.approx(6.5), pytest.approx(0.5),
    ]


def test_offset_resolved_mesh_skips_begin_mode():
    """Real PyMOL uses VERTEX == TRIANGLES == 4; only VERTEX coords may move."""
    from pymol import cgo

    content = [
        cgo.BEGIN, cgo.TRIANGLES,
        cgo.COLOR, 1.0, 0.0, 0.0,
        cgo.NORMAL, 0.0, 1.0, 0.0,
        cgo.VERTEX, 1.0, 2.0, 3.0,
        cgo.END,
    ]
    offset_cgo_vertices(content, (10.0, 0.0, 0.0))
    assert content[0] == cgo.BEGIN
    assert content[1] == cgo.TRIANGLES
    assert content[2] == cgo.COLOR
    assert content[3:6] == [pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.0)]
    assert content[6] == cgo.NORMAL
    assert content[7:10] == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(0.0)]
    assert content[11:14] == [pytest.approx(11.0), pytest.approx(2.0), pytest.approx(3.0)]
    assert content[14] == cgo.END


def test_offset_real_pymol_float_opcodes_alias_vertex_and_triangles(monkeypatch):
    """PyMOL 3: BEGIN=2, VERTEX=TRIANGLES=4.0; coords of 4.0 must not resync the walker."""
    import pymol.cgo as cgo_mod

    monkeypatch.setattr(cgo_mod, "BEGIN", 2.0)
    monkeypatch.setattr(cgo_mod, "END", 3.0)
    monkeypatch.setattr(cgo_mod, "VERTEX", 4.0)
    monkeypatch.setattr(cgo_mod, "TRIANGLES", 4.0)
    monkeypatch.setattr(cgo_mod, "NORMAL", 5.0)
    monkeypatch.setattr(cgo_mod, "COLOR", 6.0)
    monkeypatch.setattr(cgo_mod, "SPHERE", 7.0)
    monkeypatch.setattr(cgo_mod, "CYLINDER", 9.0)
    monkeypatch.setattr(cgo_mod, "CONE", 27.0)
    monkeypatch.setattr(cgo_mod, "LINEWIDTH", 10.0)
    monkeypatch.setattr(cgo_mod, "ALPHA", 25.0)

    content = [
        2.0, 4.0,
        6.0, 1.0, 0.0, 0.0,
        5.0, 0.0, 1.0, 0.0,
        4.0, 4.0, 2.0, 3.0,
        3.0,
        6.0, 0.0, 1.0, 0.0,
        7.0, 1.0, 2.0, 3.0, 0.5,
        10.0, 2.0,
        2.0, 1.0,
        4.0, 8.0, 9.0, 10.0,
        3.0,
    ]
    offset_cgo_vertices(content, (10.0, 0.0, 0.0))
    assert content[0:2] == [2.0, 4.0]
    assert content[2:6] == [6.0, 1.0, 0.0, 0.0]
    assert content[6:10] == [5.0, 0.0, 1.0, 0.0]
    assert content[10:14] == [4.0, pytest.approx(14.0), pytest.approx(2.0), pytest.approx(3.0)]
    assert content[14] == 3.0
    assert content[19:24] == [7.0, pytest.approx(11.0), pytest.approx(2.0), pytest.approx(3.0), 0.5]
    assert content[24:26] == [10.0, 2.0]
    assert content[26:28] == [2.0, 1.0]
    assert content[28:32] == [4.0, pytest.approx(18.0), pytest.approx(9.0), pytest.approx(10.0)]
