import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate

DEBUG = False

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


def _debug_plot(
    final_coords,
    poly_adjusted,
    clipped_poly,
    rect_a,
    projections,
    cx,
    cy,
    angle,
):
    if final_coords:
        fx, fy = zip(*final_coords)
        plt.fill(fx, fy, alpha=0.3, color='orange', label='final_coords')
        plt.plot(fx, fy, color='orange')
    if not poly_adjusted.is_empty:
        if poly_adjusted.geom_type == 'Polygon':
            px, py = poly_adjusted.exterior.xy
            plt.fill(px, py, alpha=0.3, color='blue', label='poly_adjusted')
            plt.plot(px, py, color='blue')
        elif poly_adjusted.geom_type == 'MultiPolygon':
            for p in poly_adjusted.geoms:
                if p.geom_type == 'Polygon':
                    px, py = p.exterior.xy
                    plt.fill(px, py, alpha=0.3, color='blue', label='poly_adjusted')
                    plt.plot(px, py, color='blue')
        elif poly_adjusted.geom_type == 'Point':
            plt.scatter([poly_adjusted.x], [poly_adjusted.y], color='blue', label='poly_adjusted')
        elif poly_adjusted.geom_type == 'GeometryCollection':
            for p in poly_adjusted.geoms:
                if p.geom_type == 'Polygon':
                    px, py = p.exterior.xy
                    plt.fill(px, py, alpha=0.3, color='blue', label='poly_adjusted')
                    plt.plot(px, py, color='blue')
    if not clipped_poly.is_empty:
        if clipped_poly.geom_type == 'Polygon':
            cpx, cpy = clipped_poly.exterior.xy
            plt.fill(cpx, cpy, alpha=0.3, color='green', label='clipped_poly')
            plt.plot(cpx, cpy, color='green')
        elif clipped_poly.geom_type == 'MultiPolygon':
            for p in clipped_poly.geoms:
                cpx, cpy = p.exterior.xy
                plt.fill(cpx, cpy, alpha=0.3, color='green', label='clipped_poly')
                plt.plot(cpx, cpy, color='green')
        elif clipped_poly.geom_type == 'GeometryCollection':
            for p in clipped_poly.geoms:
                if p.geom_type == 'Polygon':
                    cpx, cpy = p.exterior.xy
                    plt.fill(cpx, cpy, alpha=0.3, color='green', label='clipped_poly')
                    plt.plot(cpx, cpy, color='green')
    if not rect_a.is_empty and rect_a.geom_type == 'Polygon':
        rx, ry = rect_a.exterior.xy
        plt.fill(rx, ry, alpha=0.3, color='red', label='rect_a')
        plt.plot(rx, ry, color='red')

    # plot projections
    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    proj_x = [cx + p[0] * ux for p in projections]
    proj_y = [cy + p[0] * uy for p in projections]
    plt.scatter(proj_x, proj_y, color='purple', label='projections', zorder=5)
    for px_, py_ in zip(proj_x, proj_y):
        plt.plot([cx, px_], [cy, py_], color='purple', linewidth=0.8, linestyle='--')

    # plot original points
    orig_x = [p[1] for p in projections]
    orig_y = [p[2] for p in projections]
    plt.scatter(orig_x, orig_y, color='black', label='original_points', zorder=5)

    # plot center
    plt.scatter([cx], [cy], color='cyan', label='center', zorder=5)

    plt.legend()
    plt.axis('equal')
    os.makedirs("sample-images", exist_ok=True)
    plt.savefig(f"sample-images/debug_polygon_plot-{int(angle)}.png")
    plt.close()

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
            boundary = get_rect_boundary(x, y, ux, uy, shifted_x, shifted_y, rw, rh)
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

    if DEBUG:
        _debug_plot(final_coords, poly_adjusted, clipped_poly, rect_a, projections, cx, cy, angle)

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