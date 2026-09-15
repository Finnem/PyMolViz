"""Native PyMOL drag widget mapped onto clip-plane origin/normal."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.clip_drag import (
    CLIP_DRAG_NAME,
    apply_drag_matrix,
    homogeneous_from_ttt,
    matrix_changed,
    read_object_matrix,
    start_clip_drag,
    stop_clip_drag,
)
from pymolviz.util.view import translation_ttt


def test_apply_drag_matrix_translates_origin_and_center():
    origin, normal, center = apply_drag_matrix(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        [
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.5,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    assert origin == pytest.approx([3.0, 0.0, 0.5])
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    assert center == pytest.approx([3.0, 0.0, 0.5])


def test_apply_drag_matrix_rotates_around_visual_center():
    # 90° about Z through the visual center at origin.
    origin, normal, center = apply_drag_matrix(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        [
            0.0, -1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    assert origin == pytest.approx([0.0, 1.0, 0.0])
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    assert center == pytest.approx([0.0, 0.0, 0.0])


def test_apply_drag_matrix_rotates_keep_side_normal():
    origin, normal, _center = apply_drag_matrix(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    assert origin == pytest.approx([0.0, 0.0, 0.0])
    assert normal == pytest.approx([0.0, -1.0, 0.0])


def test_matrix_changed_detects_translation():
    identity = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    moved = list(identity)
    moved[3] = 0.2
    assert not matrix_changed(identity, identity)
    assert matrix_changed(identity, moved)


def test_start_clip_drag_uses_matrix_mode_without_replacing_wizard(fake_cmd):
    old = start_clip_drag(fake_cmd, (1.0, 2.0, 3.0))
    assert CLIP_DRAG_NAME in fake_cmd.objects
    assert fake_cmd.get_drag_object_name() == CLIP_DRAG_NAME
    assert fake_cmd._drag_mode == 1
    assert fake_cmd._drag_wizard == 0
    assert fake_cmd.settings.get(CLIP_DRAG_NAME, {}).get("matrix_mode") == 1
    assert ("spheres", CLIP_DRAG_NAME) in fake_cmd._shown
    matrix = read_object_matrix(fake_cmd, CLIP_DRAG_NAME)
    assert matrix[0] == pytest.approx(1.0)
    stop_clip_drag(fake_cmd, button_mode=old)
    assert fake_cmd.get_drag_object_name() == ""
    assert CLIP_DRAG_NAME not in fake_cmd.objects


def test_start_clip_drag_then_ttt_moves_read_matrix(fake_cmd):
    start_clip_drag(fake_cmd, (0.0, 0.0, 0.0))
    fake_cmd.set_object_ttt(CLIP_DRAG_NAME, translation_ttt((4.0, 0.0, 1.0)))
    matrix = read_object_matrix(fake_cmd, CLIP_DRAG_NAME)
    origin, normal, _center = apply_drag_matrix(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), matrix,
    )
    assert origin == pytest.approx([4.0, 0.0, 1.0])
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    stop_clip_drag(fake_cmd)


def test_homogeneous_from_ttt_reads_post_translation():
    matrix = homogeneous_from_ttt(translation_ttt((2.0, 0.0, -1.0)))
    assert matrix[3] == pytest.approx(2.0)
    assert matrix[7] == pytest.approx(0.0)
    assert matrix[11] == pytest.approx(-1.0)


def test_read_object_matrix_follows_ttt_even_if_get_object_matrix_is_none(fake_cmd):
    start_clip_drag(fake_cmd, (0.0, 0.0, 0.0))
    fake_cmd.set_object_ttt(CLIP_DRAG_NAME, translation_ttt((0.0, 3.0, 0.0)))
    fake_cmd.get_object_matrix = lambda *args, **kwargs: None
    matrix = read_object_matrix(fake_cmd, CLIP_DRAG_NAME)
    origin, normal, _center = apply_drag_matrix(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), matrix,
    )
    assert origin == pytest.approx([0.0, 3.0, 0.0])
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    stop_clip_drag(fake_cmd)


def test_start_clip_drag_second_attach_sees_editing_button_mode(fake_cmd):
    fake_cmd.set("button_mode", 0)
    first = start_clip_drag(fake_cmd, (0.0, 0.0, 0.0))
    assert first == 0
    second = start_clip_drag(fake_cmd, (1.0, 0.0, 0.0))
    assert second == 1
    stop_clip_drag(fake_cmd, button_mode=first)
    assert fake_cmd.get("button_mode") == 0
