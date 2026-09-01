"""Small PyMOL cmd wrappers used by wizards and interactive tools."""

from .view import translation_ttt


def restore_view(cmd_, view):
    """Re-apply a get_view() tuple without animation when possible."""
    if view is None:
        return
    try:
        cmd_.set_view(view, animate=0)
    except TypeError:
        cmd_.set_view(view)


def load_cgo_no_zoom(cmd_, cgo, name, state=1):
    """Load a CGO without auto-zooming the camera onto it."""
    from .cgo import resolve_cgo_tokens

    if any(isinstance(entry, str) for entry in cgo):
        cgo = resolve_cgo_tokens(cgo)
    view = cmd_.get_view()
    try:
        cmd_.load_cgo(cgo, name, state, zoom=0)
    except TypeError:
        loadable = getattr(cmd_, "loadable", None)
        if loadable is not None:
            try:
                cmd_.load_object(loadable.cgo, cgo, name, zoom=0)
            except TypeError:
                cmd_.load_object(loadable.cgo, cgo, name)
        else:
            cmd_.load_cgo(cgo, name, state)
    restore_view(cmd_, view)


def replace_cgo_no_zoom(cmd_, cgo, name, state=1):
    """Replace an existing CGO in place (no delete, no view save/restore)."""
    from .cgo import resolve_cgo_tokens

    if any(isinstance(entry, str) for entry in cgo):
        cgo = resolve_cgo_tokens(cgo)
    try:
        cmd_.load_cgo(cgo, name, state, zoom=0)
    except TypeError:
        loadable = getattr(cmd_, "loadable", None)
        if loadable is not None:
            try:
                cmd_.load_object(loadable.cgo, cgo, name, zoom=0)
            except TypeError:
                cmd_.load_object(loadable.cgo, cgo, name)
        else:
            cmd_.load_cgo(cgo, name, state)


def set_cgo_transparency(cmd_, name, alpha=1.0):
    """Apply object-level CGO opacity. PyMOL ignores ALPHA on SPHERE/CYLINDER."""
    transparency = max(0.0, min(1.0, 1.0 - float(alpha)))
    try:
        cmd_.set("cgo_transparency", transparency, name)
    except Exception:
        pass


WIZARD_EPHEMERAL_NAMES = (
    "pmv_camera_center",
    "_pmv_cent_tmp",
)
WIZARD_PREVIEW_PREFIXES = (
    "_pmv_prev_",
    "_pmv_sph_",
    "_pmv_box_",
    "_pmv_pt_",
    "_pmv_box_mk_",
    "_pmv_arr_",
    "_pmv_arr_mk_",
)
WIZARD_EPHEMERAL_PREFIXES = WIZARD_PREVIEW_PREFIXES + (
    "_pmv_cam_cb",
)


def purge_preview_objects(cmd_):
    """Drop uncommitted builder preview CGOs only."""
    purge_objects(cmd_, prefixes=WIZARD_PREVIEW_PREFIXES)


def purge_ephemeral_wizard_objects(cmd_):
    """Remove all wizard UI / preview CGOs (wizard not running)."""
    purge_objects(cmd_, names=WIZARD_EPHEMERAL_NAMES, prefixes=WIZARD_EPHEMERAL_PREFIXES)


def purge_objects(cmd_, names=(), prefixes=()):
    """Delete objects by exact name and/or name prefix."""
    try:
        existing = list(cmd_.get_names("objects"))
    except Exception:
        existing = list(names)
    targets = set(names)
    for obj_name in existing:
        if obj_name in targets or any(obj_name.startswith(p) for p in prefixes):
            try:
                cmd_.delete(obj_name)
            except Exception:
                pass


def place_object(cmd_, name, center):
    """Move an object with an identity+translation TTT."""
    cmd_.set_object_ttt(name, translation_ttt(center))


def set_button_action(cmd_, button, modifier, action):
    try:
        cmd_.button(button, modifier, action)
    except Exception:
        pass


def extend_cmd(cmd_, name, func):
    """Register a command for the CLI and as ``cmd.<name>`` (PyMOL 3 omits the attribute)."""
    cmd_.extend(name, func)
    try:
        setattr(cmd_, name, func)
    except Exception:
        pass


def center_on(cmd_, sele, animate=-1):
    """cmd.center with a fallback when animate= is unsupported."""
    try:
        cmd_.center(sele, animate=animate)
        return True
    except Exception:
        try:
            cmd_.center(sele)
            return True
        except Exception:
            return False


def center_on_point(cmd_, pos, tmp_name="_pmv_cent_tmp", animate=-1):
    """Center on a world-space point via a temporary pseudoatom."""
    try:
        cmd_.pseudoatom(tmp_name, pos=list(pos))
        center_on(cmd_, tmp_name, animate=animate)
    finally:
        try:
            cmd_.delete(tmp_name)
        except Exception:
            pass
