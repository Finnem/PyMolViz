"""Isovalue list + transfer-function stop helpers for Field Visuals."""

from __future__ import annotations

from typing import List, Optional


def normalize_isovalues(entries, default_level=0.0, default_color=None, default_side=1) -> list:
    if entries is None:
        entries = []
    if isinstance(entries, (int, float)):
        entries = [entries]
    out = []
    for item in entries:
        if isinstance(item, (int, float)):
            item = {"value": float(item)}
        if not isinstance(item, dict):
            continue
        side = item.get("side", default_side)
        try:
            side = int(side)
        except (TypeError, ValueError):
            side = int(default_side)
        if side not in (-1, 1):
            side = 1 if side >= 0 else -1
        opacity = item.get("opacity")
        if opacity is None:
            opacity = item.get("transparency")
            if opacity is not None:
                try:
                    opacity = 1.0 - float(opacity)
                except (TypeError, ValueError):
                    opacity = 1.0
            else:
                opacity = 1.0
        enabled = item.get("enabled", True)
        out.append({
            "value": float(item.get("value", default_level)),
            "color": item.get("color", default_color),
            "opacity": float(opacity),
            "enabled": bool(enabled),
            "side": side,
        })
    if not out:
        out.append({
            "value": float(default_level),
            "color": default_color,
            "opacity": 1.0,
            "enabled": True,
            "side": int(default_side),
        })
    return out


def primary_isovalue(entries, default_level=0.0) -> float:
    for item in normalize_isovalues(entries, default_level=default_level):
        if item.get("enabled", True):
            return float(item["value"])
    return float(default_level)


def primary_side(entries, default_side=1) -> int:
    for item in normalize_isovalues(entries, default_side=default_side):
        if item.get("enabled", True):
            return int(item.get("side", default_side))
    return int(default_side)


def isovalues_for_side(level, side) -> list:
    """Positive / negative / both as one or two ``isovalues`` entries."""
    text = str(side or "positive").strip().lower()
    value = float(level)
    if text in ("both", "two", "+-"):
        return [
            {"value": value, "enabled": True, "side": 1},
            {"value": value, "enabled": True, "side": -1},
        ]
    if text in ("negative", "neg", "-1", "-"):
        return [{"value": value, "enabled": True, "side": -1}]
    return [{"value": value, "enabled": True, "side": 1}]


def normalize_transfer_stops(stops, default_color=None) -> list:
    out = []
    for item in stops or ():
        if isinstance(item, (int, float)):
            item = {"value": float(item)}
        if not isinstance(item, dict):
            continue
        out.append({
            "value": float(item.get("value", 0.0)),
            "opacity": float(item.get("opacity", 0.03)),
            "color": item.get("color", default_color),
        })
    return out


def stops_from_volume_ramp(clims, alphas, colors=None) -> list:
    if clims is None:
        clims = []
    if alphas is None:
        alphas = []
    clims = list(clims)
    alphas = list(alphas)
    n = min(len(clims), len(alphas) if alphas else len(clims))
    stops = []
    for i in range(n):
        color = None
        if colors is not None and i < len(colors):
            color = colors[i]
        stops.append({
            "value": float(clims[i]),
            "opacity": float(alphas[i] if i < len(alphas) else 0.03),
            "color": color,
        })
    return normalize_transfer_stops(stops)


def clim_range(mode, values, custom=None) -> Optional[List[float]]:
    """Auto / custom / symmetric-about-zero color limits."""
    text = str(mode or "auto").strip().lower()
    if text in ("custom", "manual") and custom is not None:
        try:
            lo, hi = float(custom[0]), float(custom[1])
        except (TypeError, ValueError, IndexError):
            return None
        if hi < lo:
            lo, hi = hi, lo
        return [lo, hi]
    import numpy as np

    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if text in ("symmetric", "sym", "about_zero"):
        mag = max(abs(vmin), abs(vmax), 1e-12)
        return [-mag, mag]
    return None
