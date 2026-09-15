"""Reusable colormap definition, range/normalization, and cheap field histograms.

A named preset (``RdYlBu_r``) is the colormap. Range maps field values into
0–1. They are stored separately so one colormap can color several fields.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ColorMap import (
    RANGE_MODE_AUTO,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_SYMMETRIC,
    apply_colormap_reverse,
    normalize_range_mode,
)
from .field_sample import DEFAULT_SURFACE_COLORMAP, FIELD_COLORMAPS

RGBA = Tuple[float, float, float, float]

INTERP_RGB = "linear_rgb"
INTERP_HSV = "linear_hsv"
INTERPOLATIONS = (INTERP_RGB, INTERP_HSV)

MAP_CONTINUOUS = "continuous"
MAP_DISCRETE = "discrete"
MAP_TYPES = (MAP_CONTINUOUS, MAP_DISCRETE)

OOR_CLAMP = "clamp"
OOR_TRANSPARENT = "transparent"
OOR_CUSTOM = "custom"
OOR_MODES = (OOR_CLAMP, OOR_TRANSPARENT, OOR_CUSTOM)

RANGE_MODE_PERCENTILE = "percentile"
RANGE_MODES = (
    RANGE_MODE_AUTO,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_SYMMETRIC,
    RANGE_MODE_PERCENTILE,
)

DEFAULT_NAN_RGBA: RGBA = (0.55, 0.55, 0.55, 1.0)
DEFAULT_LEVELS = 5
HISTOGRAM_BINS = 48
HISTOGRAM_MAX_SAMPLES = 20000
PRESET_STOP_COUNT = 5

_HISTOGRAM_CACHE: Dict[str, dict] = {}


def normalize_interpolation(mode) -> str:
    text = str(mode or INTERP_RGB).strip().lower().replace(" ", "_").replace("-", "_")
    if text in ("hsv", "linear_hsv", "hsva"):
        return INTERP_HSV
    return INTERP_RGB


def normalize_map_type(mode) -> str:
    text = str(mode or MAP_CONTINUOUS).strip().lower()
    if text in ("discrete", "binned", "quantized"):
        return MAP_DISCRETE
    return MAP_CONTINUOUS


def normalize_oor(mode) -> str:
    text = str(mode or OOR_CLAMP).strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("transparent", "hide", "nan"):
        return OOR_TRANSPARENT
    if text in ("custom", "below_above", "below/above"):
        return OOR_CUSTOM
    return OOR_CLAMP


def normalize_range_mode_full(mode) -> str:
    text = str(mode or "").strip().lower().replace("-", "_")
    if text in ("percentile", "pct", "percentiles"):
        return RANGE_MODE_PERCENTILE
    return normalize_range_mode(mode)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_rgba(value, default_alpha: float = 1.0) -> RGBA:
    if value is None:
        return (0.0, 0.0, 0.0, float(default_alpha))
    seq = list(value)
    r = _clip01(seq[0] if len(seq) > 0 else 0.0)
    g = _clip01(seq[1] if len(seq) > 1 else 0.0)
    b = _clip01(seq[2] if len(seq) > 2 else 0.0)
    a = _clip01(seq[3] if len(seq) > 3 else default_alpha)
    return (r, g, b, a)


def _rgba_tuple(value) -> List[float]:
    rgba = _as_rgba(value)
    return [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])]


@dataclass(frozen=True)
class ColorStop:
    position: float
    rgba: RGBA
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "position": float(self.position),
            "rgba": _rgba_tuple(self.rgba),
            "label": str(self.label or ""),
        }

    @classmethod
    def from_dict(cls, data) -> "ColorStop":
        if not isinstance(data, dict):
            return cls(0.0, (0.0, 0.0, 0.0, 1.0))
        return cls(
            position=_clip01(data.get("position", 0.0)),
            rgba=_as_rgba(data.get("rgba") or data.get("color") or (0, 0, 0, 1)),
            label=str(data.get("label") or ""),
        )


def _sorted_stops(stops: Sequence[ColorStop]) -> Tuple[ColorStop, ...]:
    cleaned = []
    for stop in stops or ():
        if not isinstance(stop, ColorStop):
            continue
        cleaned.append(
            ColorStop(position=_clip01(stop.position), rgba=_as_rgba(stop.rgba), label=str(stop.label or ""))
        )
    cleaned.sort(key=lambda item: (item.position, item.label))
    if not cleaned:
        cleaned = [
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ]
    return tuple(cleaned)


@dataclass(frozen=True)
class ColormapDefinition:
    """Normalized 0–1 color mapping. Independent of field units."""

    preset: Optional[str] = None
    stops: Tuple[ColorStop, ...] = ()
    interpolation: str = INTERP_RGB
    map_type: str = MAP_CONTINUOUS
    levels: int = DEFAULT_LEVELS
    nan_rgba: RGBA = DEFAULT_NAN_RGBA
    nan_transparent: bool = False
    out_of_range: str = OOR_CLAMP
    below_rgba: Optional[RGBA] = None
    above_rgba: Optional[RGBA] = None
    customized: bool = False

    def __post_init__(self):
        object.__setattr__(self, "interpolation", normalize_interpolation(self.interpolation))
        object.__setattr__(self, "map_type", normalize_map_type(self.map_type))
        object.__setattr__(self, "out_of_range", normalize_oor(self.out_of_range))
        object.__setattr__(self, "levels", max(2, int(self.levels or DEFAULT_LEVELS)))
        object.__setattr__(self, "nan_rgba", _as_rgba(self.nan_rgba))
        if self.below_rgba is not None:
            object.__setattr__(self, "below_rgba", _as_rgba(self.below_rgba))
        if self.above_rgba is not None:
            object.__setattr__(self, "above_rgba", _as_rgba(self.above_rgba))
        object.__setattr__(self, "stops", _sorted_stops(self.stops))
        preset = str(self.preset).strip() if self.preset else None
        object.__setattr__(self, "preset", preset or None)

    @property
    def display_name(self) -> str:
        if self.customized:
            return "Custom"
        return self.preset or DEFAULT_SURFACE_COLORMAP

    def to_dict(self) -> dict:
        data = {
            "preset": self.preset,
            "stops": [stop.to_dict() for stop in self.stops],
            "interpolation": self.interpolation,
            "map_type": self.map_type,
            "levels": int(self.levels),
            "nan_rgba": _rgba_tuple(self.nan_rgba),
            "nan_transparent": bool(self.nan_transparent),
            "out_of_range": self.out_of_range,
            "customized": bool(self.customized),
        }
        if self.below_rgba is not None:
            data["below_rgba"] = _rgba_tuple(self.below_rgba)
        if self.above_rgba is not None:
            data["above_rgba"] = _rgba_tuple(self.above_rgba)
        return data

    @classmethod
    def from_dict(cls, data) -> "ColormapDefinition":
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return definition_from_preset(data)
        if not isinstance(data, dict):
            return definition_from_preset(DEFAULT_SURFACE_COLORMAP)
        stops = tuple(ColorStop.from_dict(item) for item in (data.get("stops") or ()))
        return cls(
            preset=data.get("preset"),
            stops=stops,
            interpolation=data.get("interpolation", INTERP_RGB),
            map_type=data.get("map_type", MAP_CONTINUOUS),
            levels=data.get("levels", DEFAULT_LEVELS),
            nan_rgba=_as_rgba(data.get("nan_rgba") or DEFAULT_NAN_RGBA),
            nan_transparent=bool(data.get("nan_transparent", False)),
            out_of_range=data.get("out_of_range", OOR_CLAMP),
            below_rgba=_as_rgba(data["below_rgba"]) if data.get("below_rgba") is not None else None,
            above_rgba=_as_rgba(data["above_rgba"]) if data.get("above_rgba") is not None else None,
            customized=bool(data.get("customized", bool(stops))),
        )


@dataclass(frozen=True)
class Normalization:
    """Field value → unit interval. Independent of color stops."""

    mode: str = RANGE_MODE_AUTO
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    center: Optional[float] = None
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    link_center_zero: bool = True

    def __post_init__(self):
        object.__setattr__(self, "mode", normalize_range_mode_full(self.mode))
        lo = max(0.0, min(50.0, float(self.percentile_low)))
        hi = max(50.0, min(100.0, float(self.percentile_high)))
        if hi <= lo:
            hi = min(100.0, lo + 1.0)
        object.__setattr__(self, "percentile_low", lo)
        object.__setattr__(self, "percentile_high", hi)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "vmin": None if self.vmin is None else float(self.vmin),
            "vmax": None if self.vmax is None else float(self.vmax),
            "center": None if self.center is None else float(self.center),
            "percentile_low": float(self.percentile_low),
            "percentile_high": float(self.percentile_high),
            "link_center_zero": bool(self.link_center_zero),
        }

    @classmethod
    def from_dict(cls, data) -> "Normalization":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            mode=data.get("mode", RANGE_MODE_AUTO),
            vmin=None if data.get("vmin") is None else float(data["vmin"]),
            vmax=None if data.get("vmax") is None else float(data["vmax"]),
            center=None if data.get("center") is None else float(data["center"]),
            percentile_low=data.get("percentile_low", 2.0),
            percentile_high=data.get("percentile_high", 98.0),
            link_center_zero=bool(data.get("link_center_zero", True)),
        )


@dataclass(frozen=True)
class FieldColorMapping:
    field_id: Optional[str] = None
    colormap: ColormapDefinition = field(default_factory=lambda: definition_from_preset(DEFAULT_SURFACE_COLORMAP))
    normalization: Normalization = field(default_factory=Normalization)
    title: str = ""
    units: str = ""

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "colormap": self.colormap.to_dict(),
            "normalization": self.normalization.to_dict(),
            "title": str(self.title or ""),
            "units": str(self.units or ""),
        }

    @classmethod
    def from_dict(cls, data) -> "FieldColorMapping":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            field_id=str(data["field_id"]) if data.get("field_id") else None,
            colormap=ColormapDefinition.from_dict(data.get("colormap") or {}),
            normalization=Normalization.from_dict(data.get("normalization") or {}),
            title=str(data.get("title") or ""),
            units=str(data.get("units") or ""),
        )


@dataclass(frozen=True)
class ColorbarExportSettings:
    orientation: str = "horizontal"
    width: int = 600
    height: int = 100
    dpi: int = 300
    title: str = ""
    units: str = ""
    tick_count: int = 5
    number_format: str = "auto"
    decimals: int = 2
    sig_digits: int = 3
    background: str = "transparent"
    fmt: str = "png"

    def __post_init__(self):
        orient = str(self.orientation or "horizontal").strip().lower()
        object.__setattr__(self, "orientation", "vertical" if orient.startswith("v") else "horizontal")
        fmt = str(self.fmt or "png").strip().lower().lstrip(".")
        object.__setattr__(self, "fmt", "svg" if fmt == "svg" else "png")
        bg = str(self.background or "transparent").strip().lower()
        if bg not in ("transparent", "white", "black"):
            bg = "transparent"
        object.__setattr__(self, "background", bg)
        nfmt = str(self.number_format or "auto").strip().lower()
        if nfmt not in ("auto", "fixed", "scientific"):
            nfmt = "auto"
        object.__setattr__(self, "number_format", nfmt)
        object.__setattr__(self, "width", max(32, int(self.width or 600)))
        object.__setattr__(self, "height", max(32, int(self.height or 100)))
        object.__setattr__(self, "dpi", max(36, min(1200, int(self.dpi or 300))))
        object.__setattr__(self, "tick_count", max(2, min(20, int(self.tick_count or 5))))
        object.__setattr__(self, "decimals", max(0, min(8, int(self.decimals or 2))))
        object.__setattr__(self, "sig_digits", max(1, min(8, int(self.sig_digits or 3))))

    def to_dict(self) -> dict:
        return {
            "orientation": self.orientation,
            "width": int(self.width),
            "height": int(self.height),
            "dpi": int(self.dpi),
            "title": str(self.title or ""),
            "units": str(self.units or ""),
            "tick_count": int(self.tick_count),
            "number_format": self.number_format,
            "decimals": int(self.decimals),
            "sig_digits": int(self.sig_digits),
            "background": self.background,
            "fmt": self.fmt,
        }

    @classmethod
    def from_dict(cls, data) -> "ColorbarExportSettings":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(**{key: data[key] for key in (
            "orientation", "width", "height", "dpi", "title", "units",
            "tick_count", "number_format", "decimals", "sig_digits",
            "background", "fmt",
        ) if key in data})


def definition_from_preset(name, reverse=False, n_stops: int = PRESET_STOP_COUNT) -> ColormapDefinition:
    from ..ColorMap import ColorMap

    raw = str(name or DEFAULT_SURFACE_COLORMAP)
    custom = custom_preset_definition(raw)
    if custom is None and raw.endswith("_r") and len(raw) > 2:
        custom = custom_preset_definition(raw[:-2])
        if custom is not None:
            reverse = not reverse
    if custom is not None:
        return reverse_definition(custom) if reverse else custom
    text = apply_colormap_reverse(raw, bool(reverse))
    try:
        cmap = ColorMap([0.0, 1.0], text, values_are_single_color=False)
    except Exception:
        cmap = ColorMap([0.0, 1.0], DEFAULT_SURFACE_COLORMAP, values_are_single_color=False)
        text = DEFAULT_SURFACE_COLORMAP
    n = max(2, int(n_stops))
    positions = np.linspace(0.0, 1.0, n)
    colors = np.asarray(cmap.get_color(positions), dtype=float)
    stops = tuple(
        ColorStop(float(pos), _as_rgba(row))
        for pos, row in zip(positions, colors)
    )
    return ColormapDefinition(preset=text, stops=stops, customized=False)


def reverse_definition(defn: ColormapDefinition) -> ColormapDefinition:
    if not defn.customized and defn.preset:
        return definition_from_preset(apply_colormap_reverse(defn.preset, True))
    flipped = tuple(
        ColorStop(position=_clip01(1.0 - stop.position), rgba=stop.rgba, label=stop.label)
        for stop in defn.stops
    )
    preset = apply_colormap_reverse(defn.preset, True) if defn.preset else defn.preset
    return replace(defn, preset=preset, stops=_sorted_stops(flipped), customized=True)


def _lerp_rgb(a: RGBA, b: RGBA, t: float) -> RGBA:
    t = _clip01(t)
    return tuple(float(a[i] + (b[i] - a[i]) * t) for i in range(4))  # type: ignore[return-value]


def _lerp_hsv(a: RGBA, b: RGBA, t: float) -> RGBA:
    t = _clip01(t)
    h0, s0, v0 = colorsys.rgb_to_hsv(_clip01(a[0]), _clip01(a[1]), _clip01(a[2]))
    h1, s1, v1 = colorsys.rgb_to_hsv(_clip01(b[0]), _clip01(b[1]), _clip01(b[2]))
    dh = h1 - h0
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    h = (h0 + t * dh) % 1.0
    rgb = colorsys.hsv_to_rgb(h, s0 + t * (s1 - s0), v0 + t * (v1 - v0))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(a[3] + (b[3] - a[3]) * t))


def _interpolate_stops(stops: Sequence[ColorStop], t: float, interpolation: str) -> RGBA:
    t = _clip01(t)
    if t <= stops[0].position:
        return stops[0].rgba
    if t >= stops[-1].position:
        return stops[-1].rgba
    for left, right in zip(stops, stops[1:]):
        if t <= right.position:
            span = right.position - left.position
            local = 0.0 if span < 1e-12 else (t - left.position) / span
            if interpolation == INTERP_HSV:
                return _lerp_hsv(left.rgba, right.rgba, local)
            return _lerp_rgb(left.rgba, right.rgba, local)
    return stops[-1].rgba


def _named_ramp_rgba(name: str, t) -> np.ndarray:
    from ..ColorMap import ColorMap

    cmap = ColorMap([0.0, 1.0], name, values_are_single_color=False)
    return np.asarray(cmap.get_color(t), dtype=float)


def sample_unit(defn: ColormapDefinition, t) -> np.ndarray:
    """Map unit values in [0, 1] to RGBA. ``t`` may be scalar or array."""
    arr = np.asarray(t, dtype=float)
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    out = np.zeros((flat.size, 4), dtype=float)
    finite = np.isfinite(flat)
    unit = np.clip(flat[finite], 0.0, 1.0)
    if defn.map_type == MAP_DISCRETE:
        n = max(2, int(defn.levels))
        unit = (np.floor(unit * n) + 0.5) / float(n)
        unit = np.clip(unit, 0.0, 1.0)
    if finite.any():
        if not defn.customized and defn.preset and defn.interpolation == INTERP_RGB:
            mapped = _named_ramp_rgba(defn.preset, unit)
            if mapped.ndim == 1:
                mapped = mapped.reshape(1, -1)
            if mapped.shape[1] == 3:
                extra = np.ones((mapped.shape[0], 1), dtype=float)
                mapped = np.hstack([mapped, extra])
            out[finite] = mapped[:, :4]
        else:
            lerp = _interpolate_stops
            interp = defn.interpolation
            stops = defn.stops
            for i, value in zip(np.flatnonzero(finite), unit):
                out[i] = lerp(stops, float(value), interp)
    if not finite.all():
        nan = (0.0, 0.0, 0.0, 0.0) if defn.nan_transparent else defn.nan_rgba
        out[~finite] = nan
    if scalar:
        return out[0]
    return out.reshape(arr.shape + (4,))


def ramp_rgba(defn: ColormapDefinition, n: int = 256) -> np.ndarray:
    samples = np.linspace(0.0, 1.0, max(2, int(n)))
    return np.asarray(sample_unit(defn, samples), dtype=float)


def resolve_limits(norm: Normalization, values=None) -> Optional[Tuple[float, float]]:
    """Return ``(vmin, vmax)`` or None when Auto with no data."""
    mode = normalize_range_mode_full(norm.mode)
    arr = None if values is None else np.asarray(values, dtype=float).reshape(-1)
    finite = None if arr is None else arr[np.isfinite(arr)]
    center = 0.0 if norm.link_center_zero else (0.0 if norm.center is None else float(norm.center))
    if mode == RANGE_MODE_CUSTOM:
        if norm.vmin is None or norm.vmax is None:
            return None
        lo, hi = float(norm.vmin), float(norm.vmax)
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-15:
            hi = lo + 1.0
        return (lo, hi)
    if finite is None or finite.size == 0:
        if mode == RANGE_MODE_SYMMETRIC:
            if norm.vmin is not None and norm.vmax is not None:
                mag = max(
                    abs(float(norm.vmin) - center),
                    abs(float(norm.vmax) - center),
                    1e-12,
                )
                return (center - mag, center + mag)
            mag = 1.0
            return (center - mag, center + mag)
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if mode == RANGE_MODE_PERCENTILE:
        lo = float(np.percentile(finite, norm.percentile_low))
        hi = float(np.percentile(finite, norm.percentile_high))
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-15:
            hi = lo + 1.0
        return (lo, hi)
    if mode == RANGE_MODE_SYMMETRIC:
        mag = max(abs(vmin - center), abs(vmax - center), 1e-12)
        if norm.vmin is not None and norm.vmax is not None:
            mag = max(abs(float(norm.vmin) - center), abs(float(norm.vmax) - center), mag)
        return (center - mag, center + mag)
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1.0
    return (vmin, vmax)


def map_scalars(values, defn: ColormapDefinition, vmin: float, vmax: float) -> np.ndarray:
    """Map field values to RGBA using ``defn`` and explicit limits."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros((arr.size, 4), dtype=float)
    finite = np.isfinite(arr)
    span = float(vmax) - float(vmin)
    if abs(span) < 1e-15:
        span = 1.0
    unit = np.zeros(arr.size, dtype=float)
    unit[finite] = (arr[finite] - float(vmin)) / span
    below = finite & (arr < float(vmin))
    above = finite & (arr > float(vmax))
    if defn.out_of_range == OOR_CLAMP:
        unit[finite] = np.clip(unit[finite], 0.0, 1.0)
        mapped = sample_unit(defn, unit[finite])
        out[finite] = mapped
    elif defn.out_of_range == OOR_TRANSPARENT:
        inside = finite & ~below & ~above
        mapped = sample_unit(defn, np.clip(unit[inside], 0.0, 1.0))
        out[inside] = mapped
        out[below | above] = (0.0, 0.0, 0.0, 0.0)
    else:
        inside = finite & ~below & ~above
        mapped = sample_unit(defn, np.clip(unit[inside], 0.0, 1.0))
        out[inside] = mapped
        below_c = defn.below_rgba or defn.stops[0].rgba
        above_c = defn.above_rgba or defn.stops[-1].rgba
        out[below] = below_c
        out[above] = above_c
    if not finite.all():
        nan = (0.0, 0.0, 0.0, 0.0) if defn.nan_transparent else defn.nan_rgba
        out[~finite] = nan
    return out


def mpl_colormap(defn: ColormapDefinition, name: str = "pmv_custom"):
    """Matplotlib colormap for ColorMap / ColorRamp constructors."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = [(stop.position, stop.rgba) for stop in defn.stops]
    return LinearSegmentedColormap.from_list(name or "pmv_custom", colors)


def uses_stop_sampling(defn: ColormapDefinition) -> bool:
    """True when named matplotlib lookup is not enough (custom stops / discrete / HSV / NaN)."""
    if defn.customized:
        return True
    if defn.map_type != MAP_CONTINUOUS:
        return True
    if defn.interpolation != INTERP_RGB:
        return True
    if defn.nan_transparent or defn.out_of_range != OOR_CLAMP:
        return True
    nan = _as_rgba(defn.nan_rgba)
    if nan[:3] != DEFAULT_NAN_RGBA[:3]:
        return True
    return False


def coerce_definition(colormap) -> Optional[ColormapDefinition]:
    if isinstance(colormap, ColormapDefinition):
        return colormap
    if isinstance(colormap, dict):
        return ColormapDefinition.from_dict(colormap)
    return None


def named_colormap_from_attrs(colormap=None, spec=None) -> Optional[str]:
    """Preset name stored on a visual (spec wins, then string / ColorMap name)."""
    if isinstance(spec, dict):
        preset = str(spec.get("preset") or "").strip()
        if preset:
            return preset
    if isinstance(colormap, str):
        text = colormap.strip()
        return text or None
    if colormap is None:
        return None
    name = getattr(colormap, "name", None) or getattr(colormap, "_name", None)
    if name:
        text = str(name).strip()
        if text:
            return text
    inner = getattr(colormap, "colormap", None)
    if isinstance(inner, str):
        text = inner.strip()
        if text:
            return text
    return None


def persist_colormap_attrs(colormap, spec=None):
    """``(preset_name, spec_dict_or_None)`` for mesh / visual persistence."""
    defn = ColormapDefinition.from_dict(spec) if spec else coerce_definition(colormap)
    if defn is None and isinstance(colormap, str):
        defn = custom_preset_definition(colormap)
    if defn is None:
        if colormap is None:
            return None, None
        return str(colormap), None
    name = defn.preset or "custom"
    store_spec = uses_stop_sampling(defn) or custom_preset_definition(name) is not None
    return name, (defn.to_dict() if store_spec else None)


def volume_colormap_arg(colormap=None, spec=None):
    """String name or matplotlib cmap for Volume / ColorRamp constructors."""
    defn = ColormapDefinition.from_dict(spec) if spec else coerce_definition(colormap)
    if defn is None and isinstance(colormap, str):
        defn = custom_preset_definition(colormap)
    if defn is not None:
        if uses_stop_sampling(defn) or custom_preset_definition(defn.preset or "") is not None:
            return mpl_colormap(defn)
        return defn.preset or DEFAULT_SURFACE_COLORMAP
    if colormap is not None:
        return colormap
    return DEFAULT_SURFACE_COLORMAP


def sampling_colormap(colormap=None, spec=None):
    """Value accepted by ``rgb_from_scalars``: definition, spec dict, or preset name."""
    defn = ColormapDefinition.from_dict(spec) if spec else coerce_definition(colormap)
    if defn is None:
        return colormap if colormap is not None else DEFAULT_SURFACE_COLORMAP
    if uses_stop_sampling(defn):
        return defn
    return defn.preset or DEFAULT_SURFACE_COLORMAP


def format_number(value: float, mode: str = "auto", decimals: int = 2, sig_digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return ""
    mode = str(mode or "auto").strip().lower()
    if mode == "fixed":
        text = "%.*f" % (max(0, int(decimals)), number)
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    if mode == "scientific":
        return "%.*e" % (max(1, int(sig_digits)), number)
    text = format(number, ".6g")
    if text == "-0":
        return "0"
    return text


def tick_values(vmin: float, vmax: float, count: int = 5) -> List[float]:
    n = max(2, int(count))
    lo, hi = float(vmin), float(vmax)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-15:
        hi = lo + 1.0
    return [float(v) for v in np.linspace(lo, hi, n)]


def axis_to_unit(coord, start, end) -> float:
    """Map a plot coordinate onto ``[0, 1]``. ``start`` may be greater than ``end``."""
    span = float(end) - float(start)
    if abs(span) < 1e-15:
        return 0.0
    return _clip01((float(coord) - float(start)) / span)


def unit_to_axis(unit, start, end) -> float:
    return float(start) + _clip01(unit) * (float(end) - float(start))


def data_to_unit(value, vmin, vmax) -> float:
    return axis_to_unit(value, vmin, vmax)


def unit_to_data(unit, vmin, vmax) -> float:
    return unit_to_axis(unit, vmin, vmax)


def screen_y_to_alpha(y, top, bottom) -> float:
    """Screen Y increases downward; alpha 1 is at ``top``, 0 at ``bottom``."""
    return 1.0 - axis_to_unit(y, top, bottom)


def alpha_to_screen_y(alpha, top, bottom) -> float:
    return unit_to_axis(1.0 - _clip01(alpha), top, bottom)


def clamp_range(vmin, vmax, *, gap=None) -> Tuple[float, float]:
    lo, hi = float(vmin), float(vmax)
    if hi < lo:
        lo, hi = hi, lo
    if gap is None:
        gap = max(1e-12, 1e-6 * (abs(lo) + abs(hi) + 1.0))
    if hi - lo < gap:
        hi = lo + gap
    return (lo, hi)


def colorbar_caption(title: str, units: str) -> str:
    title = str(title or "").strip()
    units = str(units or "").strip()
    if title and units:
        return "%s (%s)" % (title, units)
    return title or units


def subsample_values(values, max_samples: int = HISTOGRAM_MAX_SAMPLES) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return finite
    cap = max(32, int(max_samples))
    if n <= cap:
        return finite
    step = int(math.ceil(n / float(cap)))
    return finite[::step]


def histogram_from_values(values, bins: int = HISTOGRAM_BINS, max_samples: int = HISTOGRAM_MAX_SAMPLES) -> dict:
    samples = subsample_values(values, max_samples=max_samples)
    empty = {
        "counts": (),
        "edges": (),
        "vmin": None,
        "vmax": None,
        "mean": None,
        "std": None,
        "n": 0,
        "n_total": int(np.asarray(values).size) if values is not None else 0,
    }
    if samples.size == 0:
        return empty
    n_bins = max(8, min(128, int(bins)))
    counts, edges = np.histogram(samples, bins=n_bins)
    return {
        "counts": tuple(int(v) for v in counts),
        "edges": tuple(float(v) for v in edges),
        "vmin": float(np.min(samples)),
        "vmax": float(np.max(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "n": int(samples.size),
        "n_total": int(np.asarray(values, dtype=float).reshape(-1).size),
    }


def field_values_for_stats(field_id):
    from .field_sample import resolve_grid_from_session

    if not field_id:
        return None
    grid = resolve_grid_from_session(field_id)
    if grid is None:
        return None
    return getattr(grid, "values", None)


def field_histogram(field_id, bins: int = HISTOGRAM_BINS, max_samples: int = HISTOGRAM_MAX_SAMPLES) -> Optional[dict]:
    if not field_id:
        return None
    key = "%s:%d:%d" % (field_id, int(bins), int(max_samples))
    cached = _HISTOGRAM_CACHE.get(key)
    if cached is not None:
        return cached
    values = field_values_for_stats(field_id)
    if values is None:
        return None
    payload = histogram_from_values(values, bins=bins, max_samples=max_samples)
    _HISTOGRAM_CACHE[key] = payload
    return payload


def clear_histogram_cache(field_id=None) -> None:
    if field_id is None:
        _HISTOGRAM_CACHE.clear()
        return
    prefix = str(field_id) + ":"
    for key in list(_HISTOGRAM_CACHE):
        if key.startswith(prefix):
            _HISTOGRAM_CACHE.pop(key, None)


def field_units(field_id) -> str:
    if not field_id:
        return ""
    try:
        from .field_sample import resolve_field_from_session

        field = resolve_field_from_session(field_id)
    except Exception:
        return ""
    units = getattr(field, "units", None) if field is not None else None
    return str(units) if units else ""


def field_title(field_id) -> str:
    if not field_id:
        return ""
    try:
        from .field_sample import field_label, resolve_field_from_session

        field = resolve_field_from_session(field_id)
        return field_label(field) or ""
    except Exception:
        return ""


def mapping_from_name(
    name,
    *,
    reverse=False,
    range_mode=RANGE_MODE_AUTO,
    clims=None,
    field_id=None,
    spec=None,
) -> FieldColorMapping:
    if spec:
        defn = ColormapDefinition.from_dict(spec)
    else:
        defn = definition_from_preset(name, reverse=reverse)
    vmin = vmax = None
    if clims is not None:
        try:
            vmin, vmax = float(clims[0]), float(clims[1])
        except (TypeError, ValueError, IndexError):
            vmin = vmax = None
    return FieldColorMapping(
        field_id=str(field_id) if field_id else None,
        colormap=defn,
        normalization=Normalization(
            mode=range_mode,
            vmin=vmin,
            vmax=vmax,
            link_center_zero=True,
        ),
        title=field_title(field_id),
        units=field_units(field_id),
    )


def colormap_for_sampling(name=None, spec=None) -> object:
    """Value accepted by ``rgb_from_scalars``: definition, spec dict, or a preset name."""
    return sampling_colormap(name, spec)


def _presets_path() -> str:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = os.path.join(xdg, "pymolviz")
    else:
        base = os.path.join(os.path.expanduser("~"), ".config", "pymolviz")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "colormap_presets.json")


def load_custom_presets() -> List[dict]:
    path = _presets_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []
    entries = payload.get("presets") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    out = []
    for item in entries:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        try:
            defn = ColormapDefinition.from_dict(item.get("definition") or item)
        except Exception:
            continue
        out.append({"name": str(item["name"]), "definition": defn.to_dict()})
    return out


def save_custom_preset(name: str, defn: ColormapDefinition) -> None:
    name = str(name or "").strip()
    if not name:
        return
    entries = [item for item in load_custom_presets() if item.get("name") != name]
    payload = {"name": name, "definition": replace(defn, customized=True, preset=name).to_dict()}
    entries.insert(0, payload)
    path = _presets_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"presets": entries}, handle, indent=2)


def delete_custom_preset(name: str) -> bool:
    name = str(name or "").strip()
    if not name:
        return False
    entries = load_custom_presets()
    kept = [item for item in entries if item.get("name") != name]
    if len(kept) == len(entries):
        return False
    path = _presets_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"presets": kept}, handle, indent=2)
    return True


def create_custom_preset(defn: Optional[ColormapDefinition] = None, base: str = "Custom") -> str:
    """Save a new custom preset and return its unique name."""
    name = unused_custom_preset_name(base)
    source = defn if defn is not None else definition_from_preset(DEFAULT_SURFACE_COLORMAP)
    save_custom_preset(name, source)
    return name


def custom_colormap_rows() -> List[dict]:
    rows = []
    for item in load_custom_presets():
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        rows.append({"name": name, "definition": item.get("definition") or {}})
    return rows


def mapping_for_custom_preset(name: str) -> Optional[FieldColorMapping]:
    defn = custom_preset_definition(name)
    if defn is None:
        return None
    return FieldColorMapping(colormap=replace(defn, customized=True, preset=str(name)))


SIMILAR_RAMP_SAMPLES = 64
SIMILAR_RAMP_COLOR_MAX = 0.06
SIMILAR_RAMP_ALPHA_MAX = 0.06
SIMILAR_STOP_POS = 0.03
HIGHLIGHT_COLOR = 0.02
HIGHLIGHT_ALPHA = 0.02
HIGHLIGHT_POS = 0.01

CHOICE_USE_EXISTING = "use_existing"
CHOICE_KEEP_NEW = "keep_new"
CHOICE_CANCEL = "cancel"


@dataclass(frozen=True)
class ColormapStopDiff:
    kind: str
    position_new: Optional[float]
    position_existing: Optional[float]
    rgba_new: Optional[RGBA]
    rgba_existing: Optional[RGBA]
    color_delta: float
    alpha_delta: float
    highlight: bool


@dataclass(frozen=True)
class ColormapSimilarity:
    color_max: float
    color_mean: float
    alpha_max: float
    alpha_mean: float
    stop_diffs: Tuple[ColormapStopDiff, ...]
    similar: bool


@dataclass(frozen=True)
class SimilarColormapMatch:
    name: str
    definition: ColormapDefinition
    similarity: ColormapSimilarity


@dataclass(frozen=True)
class SimilarColormapChoice:
    action: str
    name: Optional[str] = None


def _rgba_delta(a: Optional[RGBA], b: Optional[RGBA]) -> Tuple[float, float]:
    if a is None or b is None:
        return (1.0, 1.0)
    color = max(abs(float(a[i]) - float(b[i])) for i in range(3))
    alpha = abs(float(a[3]) - float(b[3]))
    return (color, alpha)


def pair_colormap_stops(
    new_stops: Sequence[ColorStop],
    existing_stops: Sequence[ColorStop],
    pos_tol: float = SIMILAR_STOP_POS,
) -> Tuple[ColormapStopDiff, ...]:
    """Greedy position matching used for similarity and the diff table."""
    left = list(new_stops or ())
    right = list(existing_stops or ())
    used = set()
    rows: List[ColormapStopDiff] = []
    for stop in left:
        best_i = None
        best_d = 1e9
        for index, other in enumerate(right):
            if index in used:
                continue
            dist = abs(float(stop.position) - float(other.position))
            if dist < best_d:
                best_d = dist
                best_i = index
        if best_i is not None and best_d <= float(pos_tol):
            used.add(best_i)
            other = right[best_i]
            color_d, alpha_d = _rgba_delta(stop.rgba, other.rgba)
            highlight = (
                color_d > HIGHLIGHT_COLOR
                or alpha_d > HIGHLIGHT_ALPHA
                or abs(float(stop.position) - float(other.position)) > HIGHLIGHT_POS
            )
            rows.append(
                ColormapStopDiff(
                    kind="match",
                    position_new=float(stop.position),
                    position_existing=float(other.position),
                    rgba_new=stop.rgba,
                    rgba_existing=other.rgba,
                    color_delta=color_d,
                    alpha_delta=alpha_d,
                    highlight=highlight,
                )
            )
        else:
            rows.append(
                ColormapStopDiff(
                    kind="extra_new",
                    position_new=float(stop.position),
                    position_existing=None,
                    rgba_new=stop.rgba,
                    rgba_existing=None,
                    color_delta=1.0,
                    alpha_delta=1.0,
                    highlight=True,
                )
            )
    for index, other in enumerate(right):
        if index in used:
            continue
        rows.append(
            ColormapStopDiff(
                kind="extra_existing",
                position_new=None,
                position_existing=float(other.position),
                rgba_new=None,
                rgba_existing=other.rgba,
                color_delta=1.0,
                alpha_delta=1.0,
                highlight=True,
            )
        )
    rows.sort(
        key=lambda row: (
            row.position_new if row.position_new is not None else row.position_existing or 0.0,
            row.kind,
        )
    )
    return tuple(rows)


def colormap_similarity(new: ColormapDefinition, existing: ColormapDefinition, n: int = SIMILAR_RAMP_SAMPLES) -> ColormapSimilarity:
    """Compare interpolated RGBA ramps and aligned color/alpha stops."""
    left_defn = replace(new, customized=True) if new.stops else new
    right_defn = replace(existing, customized=True) if existing.stops else existing
    left = ramp_rgba(left_defn, n=n)
    right = ramp_rgba(right_defn, n=n)
    delta = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    color = delta[:, :3]
    alpha = delta[:, 3]
    color_max = float(color.max()) if color.size else 0.0
    color_mean = float(color.mean()) if color.size else 0.0
    alpha_max = float(alpha.max()) if alpha.size else 0.0
    alpha_mean = float(alpha.mean()) if alpha.size else 0.0
    diffs = pair_colormap_stops(new.stops, existing.stops)
    extras = sum(1 for row in diffs if row.kind != "match")
    matched = [row for row in diffs if row.kind == "match"]
    stops_close = extras == 0 and all(
        row.color_delta <= SIMILAR_RAMP_COLOR_MAX
        and row.alpha_delta <= SIMILAR_RAMP_ALPHA_MAX
        and abs((row.position_new or 0.0) - (row.position_existing or 0.0)) <= SIMILAR_STOP_POS
        for row in matched
    )
    ramp_close = color_max <= SIMILAR_RAMP_COLOR_MAX and alpha_max <= SIMILAR_RAMP_ALPHA_MAX
    similar = bool(ramp_close or (stops_close and matched))
    return ColormapSimilarity(
        color_max=color_max,
        color_mean=color_mean,
        alpha_max=alpha_max,
        alpha_mean=alpha_mean,
        stop_diffs=diffs,
        similar=similar,
    )


def closest_similar_custom_colormap(
    defn: ColormapDefinition,
    exclude_name: Optional[str] = None,
) -> Optional[SimilarColormapMatch]:
    """Nearest highly similar saved custom colormap, if any."""
    skip = str(exclude_name or "").strip()
    best = None
    best_score = 1e9
    for item in load_custom_presets():
        name = str(item.get("name") or "").strip()
        if not name or name == skip:
            continue
        other = ColormapDefinition.from_dict(item.get("definition") or {})
        similarity = colormap_similarity(defn, other)
        if not similarity.similar:
            continue
        score = max(similarity.color_max, similarity.alpha_max)
        if score < best_score:
            best_score = score
            best = SimilarColormapMatch(name=name, definition=other, similarity=similarity)
    return best


def builtin_preset_names() -> Tuple[str, ...]:
    return tuple(FIELD_COLORMAPS)


def custom_preset_names() -> List[str]:
    return [str(item["name"]) for item in load_custom_presets() if item.get("name")]


def is_custom_preset_name(name) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    return text in set(custom_preset_names())


def custom_preset_definition(name) -> Optional[ColormapDefinition]:
    text = str(name or "").strip()
    if not text:
        return None
    for item in load_custom_presets():
        if str(item.get("name") or "") == text:
            return ColormapDefinition.from_dict(item.get("definition") or {})
    return None


def unused_custom_preset_name(base: str = "Custom") -> str:
    used = set(builtin_preset_names())
    used.update(custom_preset_names())
    stem = str(base or "Custom").strip() or "Custom"
    n = 1
    while True:
        candidate = "%s %d" % (stem, n)
        if candidate not in used:
            return candidate
        n += 1


def listed_preset_names() -> List[str]:
    names = list(FIELD_COLORMAPS)
    for label in custom_preset_names():
        if label and label not in names:
            names.append(label)
    return names
