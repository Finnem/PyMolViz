"""Screen-space pick_atom: shape tests, no golden geometry dumps."""

from pymolviz.wizards.pick import pick_atom


class _Widget:
    fb_scale = 1.0

    def width(self):
        return 400

    def height(self):
        return 400


def _view():
    # Identity rotation, camera origin z=50, model origin at 0.
    v = [0.0] * 18
    v[0] = v[4] = v[8] = 1.0
    v[11] = 50.0
    v[15] = 1.0
    v[16] = 200.0
    return v


def test_pick_atom_returns_nearest_in_screen_pixels():
    view = _view()
    coords = [
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ]
    hit = pick_atom(view, coords, _Widget(), 200.0, 200.0, (400.0, 400.0), 20.0, 1)
    assert hit is not None
    pos, sele = hit
    assert pos == (0.0, 0.0, 0.0)
    assert sele is None


def test_pick_atom_empty_coords():
    assert pick_atom(_view(), [], _Widget(), 200.0, 200.0, (400.0, 400.0), 20.0, 1) is None


def test_pick_atom_uses_ids_when_provided():
    coords = [[0.0, 0.0, 0.0]]
    ids = [("prot", 7)]
    hit = pick_atom(_view(), coords, _Widget(), 200.0, 200.0, (400.0, 400.0), 20.0, 1, ids=ids)
    assert hit is not None
    assert hit[1] == "(prot)`7"


def test_click_ray_points_at_screen_center_passes_near_view_center():
    from pymolviz.util.view import click_ray_points, screen_center

    view = _view()
    origin = screen_center(view)
    points = click_ray_points(view, 200.0, 200.0, 400.0, 400.0, 20.0, n=8)
    assert len(points) == 8
    # Identity view: the screen-center ray is the z-axis through the view origin.
    for point in points:
        assert abs(point[0] - origin[0]) < 1e-6
        assert abs(point[1] - origin[1]) < 1e-6
    assert points[0][2] != points[-1][2]


def test_snap_get_coords_uses_ray_bounds_not_all_visible():
    from pymolviz.wizards.camera_center import CameraCenterSphere
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, name="CA"))
    queries = []
    orig = cmd.get_coords

    def recorded(sele, state=1):
        queries.append(sele)
        return orig(sele, state)

    cmd.get_coords = recorded
    sphere = CameraCenterSphere(cmd)
    queries.clear()
    sphere._pending_snap = (_Widget(), 200.0, 200.0)
    sphere._apply_pending_snap()
    assert queries
    assert all("x >" in query and "x <" in query for query in queries)
    assert all("visible and enabled" in query for query in queries)
    assert not any(query.strip("() ") in ("visible", "visible and enabled", "all") for query in queries)


def test_click_ray_selection_is_bounded_visible_box():
    from pymolviz.util.view import click_ray_selection

    expr = click_ray_selection(_view(), 200.0, 200.0, 400.0, 400.0, 20.0)
    assert "visible and enabled" in expr
    assert "x >" in expr and "x <" in expr
    assert "y >" in expr and "z >" in expr
    assert expr.strip("() ") not in ("visible", "visible and enabled", "all")


def test_record_viewer_atom_click_iterates_click_ray_not_all_visible():
    from pymolviz.wizards.last_click import last_clicked_atom, set_last_clicked_atom
    from pymolviz.wizards.pick import record_viewer_atom_click
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    cmd.get_viewport = lambda: (400.0, 400.0)
    cmd.set("orthoscopic", 1)
    cmd.set("field_of_view", 20.0)
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, name="CA"))
    cmd.add_atom(FakeAtom("prot", 2, 80.0, 0.0, 0.0, name="CB"))
    queries = []
    original = cmd.iterate

    def wrapped(sele_expr, expr, space=None):
        queries.append(str(sele_expr))
        return original(sele_expr, expr, space)

    cmd.iterate = wrapped
    set_last_clicked_atom(None)
    record_viewer_atom_click(cmd, _Widget(), 200.0, 200.0)
    assert queries
    assert all("x >" in query and "x <" in query for query in queries)
    assert all("visible and enabled" in query for query in queries)
    assert not any(
        query.strip("() ") in ("visible", "visible and enabled", "all")
        for query in queries
    )
    assert last_clicked_atom() == ("prot", 1)
    set_last_clicked_atom(None)


def test_record_viewer_atom_click_keeps_you_clicked_without_iterate():
    from pymolviz.wizards.last_click import (
        last_clicked_atom,
        last_clicked_path,
        note_click_feedback,
        set_last_clicked_atom,
    )
    from pymolviz.wizards.pick import record_viewer_atom_click
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    cmd.get_viewport = lambda: (400.0, 400.0)
    cmd.set("orthoscopic", 1)
    cmd.set("field_of_view", 20.0)
    cmd.add_atom(FakeAtom(
        "4C1D-out", 77, 0.0, 0.0, 0.0, chain="A", resn="GLY", resi="77", name="C",
    ))
    queries = []
    original = cmd.iterate

    def wrapped(sele_expr, expr, space=None):
        queries.append(str(sele_expr))
        return original(sele_expr, expr, space)

    cmd.iterate = wrapped
    set_last_clicked_atom(None)
    assert note_click_feedback(" You clicked /4C1D-out//A/GLY`77/C") is True
    record_viewer_atom_click(cmd, _Widget(), 200.0, 200.0)
    assert last_clicked_path() == "/4C1D-out//A/GLY`77/C"
    assert last_clicked_atom() in (None, ("4C1D-out", 77))
    assert queries == []
    set_last_clicked_atom(None)


def test_record_viewer_atom_click_uses_small_sele_not_all_visible():
    from pymolviz.wizards.last_click import last_clicked_atom, set_last_clicked_atom
    from pymolviz.wizards.pick import record_viewer_atom_click
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    cmd.get_viewport = lambda: (400.0, 400.0)
    cmd.set("orthoscopic", 1)
    cmd.set("field_of_view", 20.0)
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, name="CA", chain="A", resi="1"))
    for i in range(2, 9):
        cmd.add_atom(FakeAtom("prot", i, 80.0, float(i), 0.0, name="C", chain="A", resi="1"))
    cmd.select("sele", 'object "prot"')
    queries = []
    original = cmd.iterate

    def wrapped(sele_expr, expr, space=None):
        queries.append(str(sele_expr))
        return original(sele_expr, expr, space)

    cmd.iterate = wrapped
    set_last_clicked_atom(None)
    record_viewer_atom_click(cmd, _Widget(), 200.0, 200.0)
    assert last_clicked_atom() == ("prot", 1)
    assert queries
    assert any("(sele)" in query or query.strip() == "sele" for query in queries)
    assert not any(
        query.strip("() ") in ("visible", "visible and enabled", "all")
        for query in queries
    )
    set_last_clicked_atom(None)


def test_middle_click_does_not_record_atom_on_left_click():
    import inspect

    from pymolviz.wizards import middle_click

    installed = inspect.getsource(middle_click.install_middle_click_filter)
    assert "_queue_viewer_atom_click" not in installed
    assert "record_viewer_atom_click" not in installed
    assert "LeftButton" not in installed
    assert "MiddleButton" in installed


def test_idle_selection_poll_ok_uses_viewer_rect_when_widgetAt_misses():
    import inspect

    import pymolviz.wizards.pick as pick
    from pymolviz.wizards.pick import idle_selection_poll_ok

    src = inspect.getsource(idle_selection_poll_ok)
    helper = inspect.getsource(pick._cursor_in_widget)
    assert "_cursor_in_widget" in src
    assert "widgetAt" in src
    assert "under is None" in src
    assert "mapFromGlobal" in helper


def test_follow_view_does_not_fetch_coords():
    from pymolviz.wizards.camera_center import CameraCenterSphere
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    calls = {"n": 0}
    orig = cmd.get_coords

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    cmd.get_coords = counted
    sphere = CameraCenterSphere(cmd)
    assert calls["n"] == 0
    view = list(cmd._view)
    view[12] = 1.5
    sphere.follow_view(view)
    assert calls["n"] == 0


def test_camera_center_loads_cgo_and_updates_ttt():
    from pymolviz.util.pymol_helpers import CAMERA_CENTER_NAME
    from pymolviz.util.view import screen_center, translation_ttt
    from pymolviz.wizards.camera_center import CameraCenterSphere
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    sphere = CameraCenterSphere(cmd)
    assert CAMERA_CENTER_NAME in cmd.objects
    assert cmd.objects[CAMERA_CENTER_NAME]
    first = list(cmd.settings[CAMERA_CENTER_NAME]["_ttt"])
    assert first == translation_ttt(screen_center(cmd._view))

    view = list(cmd._view)
    view[12] = 3.0
    sphere.follow_view(view)
    moved = cmd.settings[CAMERA_CENTER_NAME]["_ttt"]
    assert moved == translation_ttt(screen_center(view))
    assert moved != first
    from pymol.cgo import CYLINDER, VERTEX

    tokens = cmd.objects[CAMERA_CENTER_NAME]
    assert tokens.count(CYLINDER) == 0
    assert tokens.count(VERTEX) == 24


def test_ensure_object_recreates_deleted_cgo():
    from pymolviz.util.pymol_helpers import CAMERA_CENTER_NAME
    from pymolviz.wizards.camera_center import CameraCenterSphere
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd._view = _view()
    sphere = CameraCenterSphere(cmd)
    cmd.delete(CAMERA_CENTER_NAME)
    assert CAMERA_CENTER_NAME not in cmd.objects
    sphere.ensure_object()
    assert CAMERA_CENTER_NAME in cmd.objects
    assert cmd.objects[CAMERA_CENTER_NAME]


def test_pointer_over_viewer_true_without_qt_app():
    from pymolviz.wizards.pick import idle_selection_poll_ok, pointer_over_viewer

    assert pointer_over_viewer() is True
    assert idle_selection_poll_ok() is True
    assert idle_selection_poll_ok(page=object()) is True


def test_stacked_tool_windows_raise_color_picker_last():
    from pymolviz.wizards.pick import stacked_tool_windows

    class _Win:
        def __init__(self, last=False):
            self._pmv_raise_last = last

    wizard = _Win(False)
    picker = _Win(True)
    assert stacked_tool_windows([wizard, picker]) == [wizard, picker]
    assert stacked_tool_windows([picker, wizard]) == [wizard, picker]


def test_configure_tool_window_stays_on_top_without_qt_tool():
    import inspect

    from pymolviz.wizards.pick import configure_tool_window

    src = inspect.getsource(configure_tool_window)
    assert "WindowStaysOnTopHint" in src
    assert "_pmv_no_transient" in src
    assert "| QtCore.Qt.Tool" not in src


def test_bind_tool_window_deferred_skips_deleted_color_dialog(monkeypatch):
    import pymolviz.wizards.pick as pick

    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _QtCore:
        QTimer = _FakeTimer

    class _Sig:
        def connect(self, _fn):
            return None

    class _Dialog:
        def __init__(self):
            self._alive = True
            self._pmv_window_anchor = None
            self.destroyed = _Sig()

        def isVisible(self):
            if not self._alive:
                raise RuntimeError(
                    "wrapped C/C++ object of type QColorDialog has been deleted"
                )
            return True

        def show(self):
            pass

        def raise_(self):
            pass

        def windowHandle(self):
            if not self._alive:
                raise RuntimeError(
                    "wrapped C/C++ object of type QColorDialog has been deleted"
                )
            return None

    monkeypatch.setattr(pick, "qt_modules", lambda: (_QtCore, None, None))
    monkeypatch.setattr(pick, "find_pymol_window", lambda *_: None)
    monkeypatch.setattr(pick, "_install_raise_on_parent_activate", lambda *_: None)
    monkeypatch.setattr(
        pick, "qt_widget_alive", lambda w: w is not None and getattr(w, "_alive", True)
    )
    before = list(pick._OPEN_TOOL_WINDOWS)
    try:
        dialog = _Dialog()
        dialog._pmv_window_anchor = object()
        pick.bind_tool_window(dialog)
        dialog._alive = False
        assert pending
        pending[-1]()
    finally:
        pick._OPEN_TOOL_WINDOWS[:] = before


def test_bind_tool_window_skips_transient_when_flagged(monkeypatch):
    import pymolviz.wizards.pick as pick

    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _QtCore:
        QTimer = _FakeTimer

    class _Sig:
        def connect(self, _fn):
            return None

    class _Dialog:
        def __init__(self):
            self._pmv_no_transient = True
            self._pmv_window_anchor = None
            self.destroyed = _Sig()

        def isVisible(self):
            return True

    monkeypatch.setattr(pick, "qt_modules", lambda: (_QtCore, None, None))
    monkeypatch.setattr(pick, "find_pymol_window", lambda *_: object())
    monkeypatch.setattr(pick, "_install_raise_on_parent_activate", lambda *_: None)
    before = list(pick._OPEN_TOOL_WINDOWS)
    try:
        dialog = _Dialog()
        pick.bind_tool_window(dialog)
        assert dialog in pick._OPEN_TOOL_WINDOWS
        assert pending == []
    finally:
        pick._OPEN_TOOL_WINDOWS[:] = before


def test_overlay_window_uses_top_level():
    from pymolviz.wizards.pick import overlay_window

    class _W:
        def __init__(self, top=None):
            self._top = top if top is not None else self

        def window(self):
            return self._top

    top = _W()
    nested = _W(top)
    assert overlay_window(None) is None
    assert overlay_window(top) is top
    assert overlay_window(nested) is top

    class _Missing:
        def window(self):
            return None

    leaf = _Missing()
    assert overlay_window(leaf) is leaf


def test_configure_overlay_dialog_inherits_stay_on_top(monkeypatch):
    import types

    import pymolviz.wizards.pick as pick

    qt_core = types.SimpleNamespace(
        Qt=types.SimpleNamespace(WindowStaysOnTopHint=8),
    )
    stays = qt_core.Qt.WindowStaysOnTopHint

    class _Flags:
        def __init__(self, bits):
            self.bits = bits

        def __or__(self, other):
            other_bits = other.bits if isinstance(other, _Flags) else int(other)
            return _Flags(self.bits | other_bits)

        def __and__(self, other):
            other_bits = other.bits if isinstance(other, _Flags) else int(other)
            return self.bits & other_bits

        def __bool__(self):
            return bool(self.bits)

        def __int__(self):
            return self.bits

    class _Signal:
        def connect(self, _fn):
            return None

    class _Win:
        def __init__(self, flags=0, top=None):
            self._flags = _Flags(flags)
            self._parent = None
            self._top = top if top is not None else self
            self._pmv_raise_last = False
            self.destroyed = _Signal()

        def window(self):
            return self._top

        def windowFlags(self):
            return self._flags

        def setWindowFlags(self, flags):
            if not isinstance(flags, _Flags):
                flags = _Flags(int(flags))
            self._flags = flags

        def parentWidget(self):
            return self._parent

        def setParent(self, parent, *_args):
            self._parent = parent
            if parent is not None:
                self._top = parent.window() if hasattr(parent, "window") else parent

        def winId(self):
            return 1

        def windowHandle(self):
            return None

    monkeypatch.setattr(pick, "qt_modules", lambda: (qt_core, None, None))
    before = list(pick._OPEN_TOOL_WINDOWS)
    try:
        fields = _Win(stays)
        nested = _Win(top=fields)
        nested._parent = fields
        box = _Win()
        box._parent = nested
        pick.configure_overlay_dialog(box, nested)
        assert int(box.windowFlags() & stays)
        assert box._pmv_raise_last is True
        assert box in pick._OPEN_TOOL_WINDOWS
        assert box.parentWidget() is fields
    finally:
        pick._OPEN_TOOL_WINDOWS[:] = before


def test_format_center_xyz():
    from pymolviz.wizards.camera_center import format_center_xyz

    assert format_center_xyz(None) == "Cam center: (unavailable)"
    assert format_center_xyz((1.23456, -2.0, 3.0)) == "Cam center: 1.235, -2.000, 3.000"


def test_create_cam_center_pseudoatom_unique_names():
    from pymolviz.wizards.camera_center import create_cam_center_pseudoatom
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    first = create_cam_center_pseudoatom(cmd, (1.0, 2.0, 3.0))
    second = create_cam_center_pseudoatom(cmd, (4.0, 5.0, 6.0))
    assert first == "cam_center"
    assert second != first
    assert cmd.objects[first] == [1.0, 2.0, 3.0]
    assert cmd.objects[second] == [4.0, 5.0, 6.0]
    assert cmd.settings[first]["label"].startswith("Cam center:")
