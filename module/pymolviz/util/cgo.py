"""CGO construction helpers (requires pymol.cgo)."""

import math

from pymol.cgo import CYLINDER

from .geometries import point_on_sphere


def wireframe_sphere_cgo(center, radius, color, n_lon=6, n_lat=4, n_seg=10, line_r=0.012):
    """Thin cylinder cage approximating a wireframe sphere."""
    red, green, blue = [float(c) for c in color]
    obj = []

    def add_edge(p0, p1):
        obj.extend([CYLINDER, *p0, *p1, line_r, red, green, blue, red, green, blue])

    for i in range(n_lon):
        phi = 2.0 * math.pi * i / n_lon
        for j in range(n_seg):
            add_edge(
                point_on_sphere(center, radius, math.pi * j / n_seg, phi),
                point_on_sphere(center, radius, math.pi * (j + 1) / n_seg, phi),
            )
    for i in range(1, n_lat + 1):
        theta = math.pi * i / (n_lat + 1)
        for j in range(n_seg):
            add_edge(
                point_on_sphere(center, radius, theta, 2.0 * math.pi * j / n_seg),
                point_on_sphere(center, radius, theta, 2.0 * math.pi * (j + 1) / n_seg),
            )
    return obj
