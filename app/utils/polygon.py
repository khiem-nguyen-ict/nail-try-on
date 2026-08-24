import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate

def get_rect_boundary(start_x, start_y, dx, dy, rect_x, rect_y, rect_w, rect_h):
    if dx == 0 and dy == 0:
        return None
    t_values = []
    eps = 1e-9
    if dx > eps:
        t = (rect_x + rect_w - start_x) / dx
        if t > 0:
            y_hit = start_y + t * dy
            if rect_y <= y_hit <= rect_y + rect_h:
                t_values.append(t)
    elif dx < -eps:
        t = (rect_x - start_x) / dx
        if t > 0:
            y_hit = start_y + t * dy
            if rect_y <= y_hit <= rect_y + rect_h:
                t_values.append(t)
    if dy > eps:
        t = (rect_y + rect_h - start_y) / dy
        if t > 0:
            x_hit = start_x + t * dx
            if rect_x <= x_hit <= rect_x + rect_w:
                t_values.append(t)
    elif dy < -eps:
        t = (rect_y - start_y) / dy
        if t > 0:
            x_hit = start_x + t * dx
            if rect_x <= x_hit <= rect_x + rect_w:
                t_values.append(t)
    if t_values:
        t_max = max(t_values)
        return (start_x + t_max * dx, start_y + t_max * dy)
    return None

def compute_adjusted_points(points, cx, cy, angle, shifted_x, shifted_y, rw, rh):
    if not points or len(points) < 3:
        return [(x - shifted_x, y - shifted_y) for x, y in points]

    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    
    vx = -uy
    vy = ux

    projections = []
    v_projections = []

    for x, y in points:
        dx = x - cx
        dy = y - cy
        proj = dx * ux + dy * uy
        v_proj = dx * vx + dy * vy
        projections.append((proj, x, y, dx, dy))
        v_projections.append(v_proj)

    if not projections:
        return [(x - shifted_x, y - shifted_y) for x, y in points]

    min_proj = min(p[0] for p in projections)
    max_proj = max(p[0] for p in projections)
    cut_proj = (min_proj + max_proj) / 2

    adjusted_absolute = []
    for proj, x, y, dx, dy in projections:
        if proj > cut_proj and (dx != 0 or dy != 0):
            boundary = get_rect_boundary(cx, cy, dx, dy, shifted_x, shifted_y, rw, rh)
            if boundary:
                new_x, new_y = boundary
            else:
                new_x, new_y = x, y
        else:
            new_x, new_y = x, y
        adjusted_absolute.append((new_x, new_y))

    min_v = min(v_projections)
    max_v = max(v_projections)
    
    large_length = max(rw, rh)

    raw_box = box(-large_length, min_v, large_length, max_v)
    rect_a = rotate(raw_box, float(angle), origin=(0, 0), use_radians=False)
    rect_a = translate(rect_a, xoff=cx, yoff=cy)

    poly_adjusted = Polygon(adjusted_absolute)
    
    if not poly_adjusted.is_valid:
        poly_adjusted = poly_adjusted.buffer(0)

    clipped_poly = poly_adjusted.intersection(rect_a)

    if clipped_poly.is_empty:
        final_coords = adjusted_absolute
    elif clipped_poly.geom_type == 'Polygon':
        final_coords = list(clipped_poly.exterior.coords)[:-1]
    elif clipped_poly.geom_type == 'MultiPolygon':
        largest_poly = max(clipped_poly.geoms, key=lambda p: p.area)
        final_coords = list(largest_poly.exterior.coords)[:-1]
    else:
        final_coords = adjusted_absolute

    r = [(x - shifted_x, y - shifted_y) for x, y in final_coords]
    r = list(Polygon(r).convex_hull.exterior.coords)[:-1]
    return r

def get_nail_size(a: float, points):
    # Find the width and height of the boundary rectangle containing the polygon points,
    # aligned with the vector that has angle a.
    # The width is the side perpendicular to the vector that has angle a;
    # the height is the side parallel to the vector a.

    # Handle undefined angles
    if abs(a) > 180:
        print(f"Angle {a} falls into undefined cases!")
        return 0.0, 0.0

    # Convert angle from degrees to radians
    rad = math.radians(a)

    # Unit vector parallel to the vector with angle a
    ux = math.cos(rad)
    uy = math.sin(rad)

    # Unit vector perpendicular to the vector with angle a
    px = -uy
    py = ux

    pts = np.asarray(points, dtype=float)
    proj_parallel = pts[:, 0] * ux + pts[:, 1] * uy
    proj_perp = pts[:, 0] * px + pts[:, 1] * py

    height = float(proj_parallel.max() - proj_parallel.min())
    width = float(proj_perp.max() - proj_perp.min())

    return width, height