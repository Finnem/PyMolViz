"""Module-level picklable callback: sync collections when atom xyz changes."""

from __future__ import annotations

_last_hashes = {}


def reset_follow_state():
    _last_hashes.clear()


def register_follow(runtime=None, objects=None):
    """No-op: the module-level callback is installed in ``integration.install``."""
    return


def _coord_hash(obj, context):
    from ..points import PointUnresolvedError, iter_point_sources

    parts = []
    for src in iter_point_sources(obj):
        if not src.has_dynamic_source():
            continue
        try:
            xyz = src.resolve(context)
        except PointUnresolvedError:
            xyz = src.last_xyz or (0.0, 0.0, 0.0)
        except Exception:
            xyz = src.last_xyz or (0.0, 0.0, 0.0)
        parts.append((round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4)))
    return tuple(parts)


def pymolviz_follow_callback(*_args, **_kwargs):
    """Registered via ``cmd.load_callback``; skipped during ray-trace."""
    from ..points import has_dynamic_sources
    from .context import ResolveContext
    from .runtime import get_runtime
    from .session import all_objects

    runtime = get_runtime()
    context = ResolveContext(runtime.cmd)
    for obj in all_objects():
        if not has_dynamic_sources(obj):
            continue
        digest = _coord_hash(obj, context)
        if _last_hashes.get(obj.id) != digest:
            _last_hashes[obj.id] = digest
            try:
                runtime.sync(obj)
            except Exception:
                pass
