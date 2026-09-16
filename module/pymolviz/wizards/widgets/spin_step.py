"""1-2-5 arrow steps for continuous wizard spin boxes."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Union

# Named steps for knobs with a known physical unit.
STEP_ANGSTROM = 0.1
STEP_ANGSTROM_COARSE = 0.5
STEP_RADIUS = 0.05
STEP_SPACING = 0.05
STEP_RESOLUTION = 0.25
STEP_VDW_SCALE = 0.05
STEP_TRANSPARENCY = 0.05
STEP_PERCENT = 0.5
STEP_UNIT_POSITION = 0.01
STEP_NORMAL = 0.01

_UNBOUNDED = 1.0e6


def nice_step(target: float, *, fallback: float = 0.1) -> float:
    """Round ``target`` onto the 1-2-5 decade (0.1, 0.2, 0.5, 1, …)."""
    mag = abs(float(target))
    if not math.isfinite(mag) or mag <= 0.0:
        return float(fallback)
    exp = int(math.floor(math.log10(mag)))
    frac = mag / (10.0 ** exp)
    if frac <= 1.5:
        nice = 1.0
    elif frac <= 3.5:
        nice = 2.0
    else:
        nice = 5.0
    return nice * (10.0 ** exp)


def step_for_magnitude(value: float, *, fraction: float = 0.1, fallback: float = 0.1) -> float:
    mag = abs(float(value))
    if not math.isfinite(mag) or mag <= 0.0:
        return float(fallback)
    return nice_step(mag * float(fraction), fallback=fallback)


def step_for_values(
    values: Iterable[float],
    *,
    clicks: float = 40.0,
    fallback: float = 0.1,
) -> float:
    """Step that crosses the current value span in about ``clicks`` arrows."""
    nums = []
    for raw in values:
        try:
            num = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            nums.append(num)
    if not nums:
        return float(fallback)
    span = max(nums) - min(nums)
    mag = max(abs(v) for v in nums)
    if span >= _UNBOUNDED:
        if mag > 0.0:
            return step_for_magnitude(mag, fallback=fallback)
        return float(fallback)
    if span > 0.0:
        return nice_step(span / float(clicks), fallback=fallback)
    if mag > 0.0:
        return step_for_magnitude(mag, fallback=fallback)
    return float(fallback)


def step_for_spin_value(
    value: float,
    *,
    companions: Iterable[float] = (),
    fallback: float = 0.1,
) -> float:
    """Fine step near ``value``, never coarser than the companion span step."""
    shared = step_for_values([value, *companions], fallback=fallback)
    mag = abs(float(value)) if math.isfinite(float(value)) else 0.0
    if mag > 0.0:
        return min(shared, step_for_magnitude(mag, fallback=fallback))
    return min(shared, float(fallback))


def decimals_for_step(
    step: float,
    *,
    min_decimals: int = 0,
    max_decimals: int = 6,
) -> int:
    step = abs(float(step))
    lo = int(min_decimals)
    hi = int(max_decimals)
    if hi < lo:
        hi = lo
    if not math.isfinite(step) or step <= 0.0:
        return hi
    if step >= 1.0:
        return lo
    needed = int(math.ceil(-math.log10(step)))
    return max(lo, min(hi, needed))


def apply_spin_step(spin, step: float, *, decimals: Optional[int] = None) -> None:
    """Set a fixed arrow step (and optional decimal places)."""
    if spin is None:
        return
    _assign_step(spin, float(step), None if decimals is None else int(decimals))


def apply_magnitude_step(
    spin,
    *,
    extras: Union[Iterable[float], Callable[[], Iterable[float]], None] = None,
    min_decimals: int = 2,
    max_decimals: int = 6,
    fallback: float = 0.1,
) -> None:
    """Match one spin's arrows to its current magnitude (and optional peers)."""
    if spin is None:
        return
    try:
        value = float(spin.value())
    except Exception:
        return
    extra_vals = _extra_values(extras)
    step = step_for_spin_value(value, companions=extra_vals, fallback=fallback)
    decimals = decimals_for_step(step, min_decimals=min_decimals, max_decimals=max_decimals)
    _assign_step(spin, step, decimals)


def apply_peer_steps(
    *spins,
    min_decimals: int = 2,
    max_decimals: int = 6,
    fallback: float = 0.1,
) -> None:
    """Each spin uses a fine local step, capped by the group's span."""
    alive = [spin for spin in spins if spin is not None]
    values = []
    for spin in alive:
        try:
            values.append(float(spin.value()))
        except Exception:
            continue
    for spin in alive:
        try:
            value = float(spin.value())
        except Exception:
            continue
        step = step_for_spin_value(value, companions=values, fallback=fallback)
        decimals = decimals_for_step(
            step, min_decimals=min_decimals, max_decimals=max_decimals,
        )
        _assign_step(spin, step, decimals)


def bind_magnitude_step(
    spin,
    *,
    extras: Union[Iterable[float], Callable[[], Iterable[float]], None] = None,
    min_decimals: int = 2,
    max_decimals: int = 6,
    fallback: float = 0.1,
) -> None:
    """Keep ``spin`` arrows in scale as its value (or extras) change."""
    if spin is None:
        return

    def _sync(*_args):
        apply_magnitude_step(
            spin,
            extras=extras,
            min_decimals=min_decimals,
            max_decimals=max_decimals,
            fallback=fallback,
        )

    _connect_value_changed(spin, _sync)
    _sync()


def bind_peer_steps(
    *spins,
    min_decimals: int = 2,
    max_decimals: int = 6,
    fallback: float = 0.1,
) -> None:
    """Keep a min/max/center group in scale together."""
    alive = [spin for spin in spins if spin is not None]
    if not alive:
        return

    def _sync(*_args):
        apply_peer_steps(
            *alive,
            min_decimals=min_decimals,
            max_decimals=max_decimals,
            fallback=fallback,
        )

    for spin in alive:
        _connect_value_changed(spin, _sync)
    _sync()


def radius_step(value: float) -> float:
    """Arrow step for 0–100 Å radius sliders (matches the log-segment ranges)."""
    val = abs(float(value))
    if not math.isfinite(val) or val < 1.0:
        return 0.01
    if val < 10.0:
        return 0.1
    return 1.0


def _extra_values(extras) -> Sequence[float]:
    if extras is None:
        return ()
    if callable(extras):
        try:
            extras = extras()
        except Exception:
            return ()
    out = []
    for raw in extras or ():
        try:
            num = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def _assign_step(spin, step: float, decimals: Optional[int]) -> None:
    setter = getattr(spin, "setSingleStep", None)
    if callable(setter):
        current = getattr(spin, "singleStep", None)
        try:
            prev = float(current()) if callable(current) else None
        except Exception:
            prev = None
        if prev is None or abs(prev - step) > 1e-15:
            setter(float(step))
    if decimals is None:
        return
    dec_setter = getattr(spin, "setDecimals", None)
    if not callable(dec_setter):
        return
    dec_get = getattr(spin, "decimals", None)
    try:
        prev_dec = int(dec_get()) if callable(dec_get) else None
    except Exception:
        prev_dec = None
    if prev_dec != int(decimals):
        dec_setter(int(decimals))


def _connect_value_changed(spin, callback: Callable) -> None:
    signal = getattr(spin, "valueChanged", None)
    connect = getattr(signal, "connect", None)
    if callable(connect):
        connect(callback)
