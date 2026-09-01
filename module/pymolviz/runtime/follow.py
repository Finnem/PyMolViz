"""Move anchored visuals after a source object or atom finishes moving."""

from __future__ import annotations

_in_follow = False
_tracked = None
_last_fp = None
_base_fp = None
_SETTLE_S = 0.12
_RIGID_EPS2 = 1e-6
_RELEASE_DELAY_MS = 50
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_DEBOUNCE = None
_EVENT_FILTER = None
_FILTER_WIDGET = None
_FOLLOW_JOB = 0
_pending = set()


def reset_follow_state():
    global _in_follow, _tracked, _last_fp, _base_fp, _FOLLOW_JOB, _pending
    _in_follow = False
    _tracked = None
    _last_fp = None
    _base_fp = None
    _pending = set()
    _FOLLOW_JOB += 1
    _stop_debounce()
    try:
        from .status_busy import reset_status_busy
        reset_status_busy()
    except Exception:
        pass


def invalidate_watchlist():
    """Session objects changed; rediscover anchored sources on the next callback."""
    global _tracked
    _tracked = None


def register_follow(runtime=None, objects=None):
    """No-op: the module-level hook is installed in ``integration.install``."""
    return


def _stop_debounce():
    global _DEBOUNCE
    timer = _DEBOUNCE
    if timer is None:
        return
    try:
        timer.stop()
    except Exception:
        pass


def _teardown_event_filter():
    global _EVENT_FILTER, _FILTER_WIDGET
    widget = _FILTER_WIDGET
    filt = _EVENT_FILTER
    _EVENT_FILTER = None
    _FILTER_WIDGET = None
    if widget is not None and filt is not None:
        try:
            widget.removeEventFilter(filt)
        except Exception:
            pass


def uninstall_follow_hooks():
    _stop_debounce()
    _teardown_event_filter()
    reset_follow_state()


def _viewer_buttons_down():
    from ..wizards.pick import qt_modules

    _qt, _gui, QtWidgets = qt_modules()
    if QtWidgets is None:
        return False
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return False
        return bool(int(app.mouseButtons()))
    except Exception:
        return False


def request_follow_check(delay_ms=None):
    """After viewer interaction ends, compare source coords once."""
    from ..wizards.pick import qt_modules

    QtCore, _, _ = qt_modules()
    if QtCore is None or not hasattr(QtCore, "QTimer"):
        pymolviz_follow_callback()
        return
    global _DEBOUNCE
    timer = _DEBOUNCE
    if timer is None:
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(pymolviz_follow_callback)
        _DEBOUNCE = timer
    try:
        timer.stop()
        timer.start(int(_RELEASE_DELAY_MS if delay_ms is None else delay_ms))
    except Exception:
        pymolviz_follow_callback()


def ensure_follow_input_hook():
    """Listen for mouse-up / wheel-end on the 3D view. No periodic cmd poll."""
    from ..wizards.pick import find_viewer_widget, qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None or QtWidgets is None:
        return
    widget = find_viewer_widget(QtWidgets)
    if widget is None:
        return
    global _EVENT_FILTER, _FILTER_WIDGET
    if _FILTER_WIDGET is widget and _EVENT_FILTER is not None:
        return
    _teardown_event_filter()

    class _FollowInputFilter(QtCore.QObject):
        def eventFilter(self, watched, event):
            if watched is not widget:
                return False
            etype = event.type()
            release = getattr(QtCore.QEvent, "MouseButtonRelease", None)
            wheel = getattr(QtCore.QEvent, "Wheel", None)
            if etype == release or etype == wheel:
                request_follow_check()
            return False

    try:
        filt = _FollowInputFilter(widget)
        widget.installEventFilter(filt)
    except Exception:
        return
    _EVENT_FILTER = filt
    _FILTER_WIDGET = widget


def _source_object_name(src):
    name = getattr(src, "object", None)
    if not name:
        return None
    return str(name)


def _source_names_for(obj):
    from ..points import iter_point_sources

    names = []
    seen = set()
    for src in iter_point_sources(obj):
        if not src.has_dynamic_source():
            continue
        name = _source_object_name(src)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _refresh_watchlist():
    global _tracked
    from ..points import has_dynamic_sources
    from .session import all_objects

    _tracked = [obj for obj in all_objects() if has_dynamic_sources(obj)]


def _flat_matrix(matrix):
    if matrix is None:
        return None
    flat = []
    try:
        for item in matrix:
            try:
                flat.extend(float(v) for v in item)
            except TypeError:
                flat.append(float(item))
    except TypeError:
        return None
    if len(flat) < 16:
        return None
    return tuple(round(v, 5) for v in flat[:16])


def _object_ttt(cmd, name):
    """Object TTT for ``set_object_ttt``. Not ``get_object_matrix`` (homogenous)."""
    try:
        ttt = _flat_matrix(cmd.get_object_ttt(name))
        if ttt is not None:
            return ttt
    except Exception:
        pass
    return _IDENTITY


def _resolve_context(cmd):
    from .context import ResolveContext

    try:
        state = int(cmd.get_state()) or 1
    except Exception:
        state = 1
    return ResolveContext(cmd, state)


def _coordset_fp(cmd, name, state):
    """Untransformed object coords — camera motion does not change this."""
    coords = None
    try:
        coords = cmd.get_coordset(name, state)
    except TypeError:
        try:
            coords = cmd.get_coordset(name)
        except Exception:
            coords = None
    except Exception:
        coords = None
    if coords is None:
        return None
    try:
        flat = []
        for point in coords:
            flat.extend((round(float(point[0]), 3), round(float(point[1]), 3), round(float(point[2]), 3)))
        return tuple(flat)
    except Exception:
        return None


def _scene_fingerprint(cmd):
    context = _resolve_context(cmd)
    state = context.state
    matrices = []
    coordsets = []
    seen = set()
    for obj in _tracked or ():
        for name in _source_names_for(obj):
            if name in seen:
                continue
            seen.add(name)
            matrices.append((name, _object_ttt(cmd, name)))
            coordsets.append((name, _coordset_fp(cmd, name, state)))
    return (tuple(matrices), tuple(coordsets))


def capture_baseline(runtime=None):
    """Snapshot source coords when a visual is created, before the user moves atoms."""
    global _tracked, _last_fp, _base_fp
    _refresh_watchlist()
    if not _tracked:
        _last_fp = None
        _base_fp = None
        return
    if runtime is None:
        from .runtime import get_runtime
        runtime = get_runtime()
    fp = _scene_fingerprint(runtime.cmd)
    _last_fp = fp
    _base_fp = fp


def _same_delta(deltas):
    if not deltas:
        return False
    x0, y0, z0 = deltas[0]
    for x, y, z in deltas[1:]:
        dx, dy, dz = x - x0, y - y0, z - z0
        if dx * dx + dy * dy + dz * dz > _RIGID_EPS2:
            return False
    return True


def _set_visual_ttt(cmd, visual_name, matrix):
    if not visual_name or matrix is None:
        return
    try:
        cmd.set_object_ttt(visual_name, list(matrix))
    except Exception:
        pass


def _iter_meshes(obj):
    if type(obj).__name__ == "CGOCollection":
        return list(obj)
    return [obj]


def _nudge_mesh(mesh, context):
    from ..points import iter_point_sources

    sources = [src for src in iter_point_sources(mesh) if src.has_dynamic_source()]
    if not sources:
        return True
    if not hasattr(mesh, "shift_vertices"):
        return False
    deltas = []
    for src in sources:
        old = src.last_xyz
        try:
            new = src.resolve(context, remember=False)
        except Exception:
            return False
        if old is None:
            src._last_xyz = new
            continue
        deltas.append((src, (
            float(new[0]) - float(old[0]),
            float(new[1]) - float(old[1]),
            float(new[2]) - float(old[2]),
        ), new))
    if not deltas:
        return True
    if not _same_delta([item[1] for item in deltas]):
        return False
    delta = deltas[0][1]
    if delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] < 1e-16:
        for src, _delta, new in deltas:
            src._last_xyz = new
        return True
    try:
        mesh.shift_vertices(delta)
    except Exception:
        return False
    for src, _delta, new in deltas:
        src._last_xyz = new
    return True


def _nudge_object(runtime, obj, load=True):
    """Shift cached CGO vertices in place; optionally reload the PyMOL object."""
    context = runtime._context()
    failed = []
    for child in _iter_meshes(obj):
        if not _nudge_mesh(child, context):
            failed.append(child)
    if failed:
        for child in failed:
            try:
                if hasattr(child, "rebuild"):
                    child.rebuild(context)
                if hasattr(child, "invalidate_cgo_cache"):
                    child.invalidate_cgo_cache()
            except Exception:
                return False
    if not load:
        return True
    try:
        runtime.replace_cgo(obj)
    except Exception:
        return False
    return True


def _visual_name(runtime, obj):
    binding = runtime.bindings.get(obj.id)
    if binding is None:
        return None
    return binding.pymol_name


def _object_enabled(cmd, name):
    if not name:
        return False
    try:
        names = cmd.get_names("objects", 1)
    except TypeError:
        try:
            names = cmd.get_names("objects", enabled_only=1)
        except TypeError:
            names = None
    except Exception:
        names = None
    if names is not None:
        return str(name) in names
    try:
        value = cmd.get("enabled", name)
    except Exception:
        return True
    if isinstance(value, str):
        return value.lower() not in ("0", "off", "false", "no")
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return True


def _pending_enabled_objects(runtime):
    out = []
    for obj in _tracked or ():
        if obj.id not in _pending:
            continue
        if _object_enabled(runtime.cmd, _visual_name(runtime, obj)):
            out.append(obj)
    return out


def _load_visual(runtime, obj):
    """Reload CGO tokens and keep the visual's existing TTT (object-space vertices)."""
    visual = _visual_name(runtime, obj)
    prev_ttt = _object_ttt(runtime.cmd, visual) if visual else None
    try:
        runtime.replace_cgo(obj)
    except Exception:
        try:
            runtime.sync(obj)
        except Exception:
            return
    if visual and prev_ttt is not None and prev_ttt != _IDENTITY:
        _set_visual_ttt(runtime.cmd, visual, prev_ttt)


def _plan_position_updates(runtime, base_fp, new_fp):
    """Collect CGO reloads and TTT copies. Does not touch PyMOL objects."""
    base_mat = dict((base_fp or ((), ()))[0])
    new_mat = dict((new_fp or ((), ()))[0])
    base_cs = dict((base_fp or ((), ()))[1] or ())
    new_cs = dict((new_fp or ((), ()))[1] or ())
    reloads = []
    ttts = []
    rebuilt = False
    for obj in _tracked or ():
        binding = runtime.bindings.get(obj.id)
        visual = binding.pymol_name if binding is not None else None
        names = _source_names_for(obj)
        coords_changed = any(base_cs.get(name) != new_cs.get(name) for name in names)
        matrix_changed = any(base_mat.get(name) != new_mat.get(name) for name in names)
        source_matrix = new_mat.get(names[0]) if len(names) == 1 else None
        if not coords_changed and not matrix_changed:
            continue
        if coords_changed:
            reloads.append(obj)
            rebuilt = True
        if source_matrix is not None and matrix_changed:
            ttts.append((visual, source_matrix))
    return reloads, ttts, rebuilt


def _commit_position_updates(runtime, to_load, ttts):
    from .status_busy import end_cgo_update

    try:
        for obj in to_load:
            _load_visual(runtime, obj)
            _pending.discard(obj.id)
        for visual, source_matrix in ttts:
            _set_visual_ttt(runtime.cmd, visual, source_matrix)
    finally:
        if to_load:
            end_cgo_update()


def _apply_positions(runtime, base_fp, new_fp):
    """Follow atom edits by shifting CGO vertices; copy object TTT when the molecule moves.

    Hidden visuals skip ``load_cgo`` until they are enabled again.
    Shown reloads are scheduled after one Qt paint so a progress bar can show.
    Returns (rebuilt, deferred).
    """
    from .status_busy import begin_cgo_update, run_after_paint

    reloads, ttts, rebuilt = _plan_position_updates(runtime, base_fp, new_fp)
    hidden_ttts = []
    shown_ttts = []
    for visual, source_matrix in ttts:
        if _object_enabled(runtime.cmd, visual):
            shown_ttts.append((visual, source_matrix))
        else:
            hidden_ttts.append((visual, source_matrix))
    for visual, source_matrix in hidden_ttts:
        _set_visual_ttt(runtime.cmd, visual, source_matrix)

    to_load = []
    for obj in reloads:
        _nudge_object(runtime, obj, load=False)
        visual = _visual_name(runtime, obj)
        if _object_enabled(runtime.cmd, visual):
            _pending.discard(obj.id)
            to_load.append(obj)
        else:
            _pending.add(obj.id)
    for obj in _pending_enabled_objects(runtime):
        if obj not in to_load:
            to_load.append(obj)

    if not to_load:
        for visual, source_matrix in shown_ttts:
            _set_visual_ttt(runtime.cmd, visual, source_matrix)
        return rebuilt, False

    job = _FOLLOW_JOB
    begin_cgo_update()

    def _commit():
        global _in_follow
        from .status_busy import end_cgo_update
        try:
            if job != _FOLLOW_JOB:
                end_cgo_update()
                return
            _commit_position_updates(runtime, to_load, shown_ttts)
        except Exception:
            pass
        finally:
            if job == _FOLLOW_JOB:
                _in_follow = False

    deferred = run_after_paint(_commit)
    return rebuilt, deferred


def pymolviz_follow_callback(*_args, **_kwargs):
    """One-shot check after mouse/wheel interaction — never from draw or a poll loop."""
    global _in_follow, _tracked, _last_fp, _base_fp
    if _in_follow:
        return
    if _viewer_buttons_down():
        request_follow_check()
        return
    _in_follow = True
    deferred = False
    try:
        from .runtime import get_runtime

        runtime = get_runtime()
        if _tracked is None:
            _refresh_watchlist()
        if not _tracked:
            return

        fp = _scene_fingerprint(runtime.cmd)
        if _last_fp is None or _base_fp is None:
            _last_fp = fp
            _base_fp = fp
            return
        pending_show = bool(_pending_enabled_objects(runtime))
        if fp == _base_fp and not pending_show:
            _last_fp = fp
            return

        rebuilt, deferred = _apply_positions(runtime, _base_fp, fp)
        _last_fp = fp
        if rebuilt:
            _base_fp = fp
        else:
            _base_fp = (fp[0], _base_fp[1])
    except Exception:
        deferred = False
    finally:
        if not deferred:
            _in_follow = False
