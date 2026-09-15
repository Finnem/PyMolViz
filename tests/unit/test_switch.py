"""Switch metrics (no Qt)."""

from pymolviz.wizards.widgets.switch import (
    COMPACT_KNOB,
    COMPACT_TRACK_H,
    COMPACT_TRACK_W,
    KNOB,
    TRACK_H,
    TRACK_W,
    track_metrics,
)


def test_track_metrics_labeled_vs_compact():
    assert track_metrics() == (TRACK_W, TRACK_H, KNOB)
    assert track_metrics(False) == (TRACK_W, TRACK_H, KNOB)
    compact = track_metrics(True)
    assert compact == (COMPACT_TRACK_W, COMPACT_TRACK_H, COMPACT_KNOB)
    assert compact[0] < TRACK_W
    assert compact[1] <= TRACK_H
