"""Publication colorbar PNG/SVG from a FieldColorMapping (no Qt)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .colormap_spec import (
    ColorbarExportSettings,
    ColormapDefinition,
    FieldColorMapping,
    Normalization,
    colorbar_caption,
    format_number,
    ramp_rgba,
    resolve_limits,
    tick_values,
)


def _background_rgba(name: str):
    text = str(name or "transparent").strip().lower()
    if text == "white":
        return (1.0, 1.0, 1.0, 1.0)
    if text == "black":
        return (0.0, 0.0, 0.0, 1.0)
    return (1.0, 1.0, 1.0, 0.0)


def _tick_labels(vmin, vmax, settings: ColorbarExportSettings):
    values = tick_values(vmin, vmax, settings.tick_count)
    return [
        format_number(v, settings.number_format, settings.decimals, settings.sig_digits)
        for v in values
    ], values


def export_colorbar(
    path: str,
    mapping: FieldColorMapping,
    settings: Optional[ColorbarExportSettings] = None,
    values=None,
    limits=None,
) -> str:
    """Write ``path`` as PNG or SVG. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    settings = settings or ColorbarExportSettings()
    defn = mapping.colormap if isinstance(mapping.colormap, ColormapDefinition) else ColormapDefinition.from_dict(mapping.colormap)
    norm = mapping.normalization if isinstance(mapping.normalization, Normalization) else Normalization.from_dict(mapping.normalization)
    if limits is None:
        limits = resolve_limits(norm, values)
    if limits is None:
        limits = (0.0, 1.0)
    vmin, vmax = float(limits[0]), float(limits[1])
    n = 256
    colors = ramp_rgba(defn, n=n)
    cmap = ListedColormap(np.clip(colors, 0.0, 1.0))
    labels, ticks = _tick_labels(vmin, vmax, settings)
    caption = colorbar_caption(settings.title or mapping.title, settings.units or mapping.units)
    bg = _background_rgba(settings.background)
    transparent = settings.background == "transparent"
    horizontal = settings.orientation != "vertical"
    width_in = max(settings.width, 32) / float(settings.dpi)
    height_in = max(settings.height, 32) / float(settings.dpi)
    fig = plt.figure(figsize=(width_in, height_in), dpi=float(settings.dpi))
    fig.patch.set_facecolor(bg)
    if horizontal:
        ax = fig.add_axes([0.08, 0.38, 0.84, 0.28])
        gradient = np.linspace(0.0, 1.0, n).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=(vmin, vmax, 0.0, 1.0))
        ax.set_yticks([])
        ax.set_xlim(vmin, vmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if caption:
            ax.set_title(caption, pad=8, fontsize=11)
    else:
        ax = fig.add_axes([0.42, 0.08, 0.28, 0.84])
        gradient = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=(0.0, 1.0, vmin, vmax))
        ax.set_xticks([])
        ax.set_ylim(vmin, vmax)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        if caption:
            ax.set_ylabel(caption, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(length=3, labelsize=8)
    path = str(path)
    fig.savefig(
        path,
        dpi=float(settings.dpi),
        format=settings.fmt,
        transparent=transparent,
        facecolor=bg if not transparent else "none",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)
    return path
