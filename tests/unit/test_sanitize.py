"""PyMOL object-name sanitization and selection quoting."""

from pymolviz.Displayable import Displayable
from pymolviz.util.sanitize import (
    object_sele,
    parse_object_sele,
    quote_pymol_name,
    sanitize_pymol_string,
)


def test_sanitize_union_charset():
    assert sanitize_pymol_string("a b,c.d-e") == "a_b_c_d_e"
    assert sanitize_pymol_string("1foo") == "_1foo"
    assert sanitize_pymol_string("clean_name") == "clean_name"


def test_sanitize_idempotent():
    raw = "foo,bar-1"
    once = sanitize_pymol_string(raw)
    assert sanitize_pymol_string(once) == once
    assert once == "foo_bar_1"


def test_sanitize_none_and_empty():
    assert sanitize_pymol_string(None) is None
    assert sanitize_pymol_string("") == ""


def test_displayable_name_matches_sanitize():
    raw = "foo,bar-1"
    obj = Displayable(name=raw)
    assert obj.name == sanitize_pymol_string(raw) == "foo_bar_1"


def test_quote_pymol_name_preserves_hyphen():
    assert quote_pymol_name("1abc-A") == '"1abc-A"'


def test_quote_pymol_name_already_parenthesized():
    assert quote_pymol_name("(sele)") == "(sele)"


def test_object_sele_avoids_bare_map_name():
    assert object_sele("input_refine_001_2Fo_Fc") == 'object "input_refine_001_2Fo_Fc"'
    assert parse_object_sele('object "input_refine_001_2Fo_Fc"') == "input_refine_001_2Fo_Fc"
    assert object_sele("(sele)") == "(sele)"
