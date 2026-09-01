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


def test_pointer_over_viewer_true_without_qt_app():
    from pymolviz.wizards.pick import pointer_over_viewer

    assert pointer_over_viewer() is True
