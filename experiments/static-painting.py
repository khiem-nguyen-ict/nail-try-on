import json
import sys
import os
import colorsys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import MAX_DETECTION_DIM
from app.services.hand_detector import detect_hands
from app.services.nail_detector import detect_nails, filter_nails_by_hands
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps, ImageEnhance, ImageStat, ImageFont
import math

GAUSSIAN_BLUR=6

# Color matching defaults
COLOR_MATCH_HUE_SHIFT = 0.04       # max hue shift toward base image (0-1, fraction of hue circle)
COLOR_MATCH_SATURATION = 0.15      # how much to blend saturation toward base image (0-1)
COLOR_MATCH_BRIGHTNESS = 0.2       # how much to blend brightness toward base image (0-1)

ORIGINAL_IMAGE = "sample-images/hand-6.jpg"
REFERENCE_IMAGE = "sample-images/sample-2.png"

DEBUG_FONT_PATHS = [
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def get_debug_font(length):
    """Return a font scaled proportionally to the nail length."""
    font_size = max(12, int(length / 8))
    for path in DEBUG_FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default(size=font_size)


def get_base_image_color_profile(base_image):
    """Analyze base image and return average HSV values."""
    small = base_image.resize((64, 64), Image.Resampling.LANCZOS)
    stat = ImageStat.Stat(small)
    r_mean = stat.mean[0] / 255.0
    g_mean = stat.mean[1] / 255.0
    b_mean = stat.mean[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_mean, g_mean, b_mean)
    return {"hue": h, "saturation": s, "brightness": v}


def apply_color_matching(nail_img, base_profile):
    """Subtly adjust nail HSV per-pixel to harmonize with base image."""
    if nail_img.mode != "RGBA":
        nail_img = nail_img.convert("RGBA")

    r, g, b, a = nail_img.split()
    rgb_img = Image.merge("RGB", (r, g, b))
    hsv_img = rgb_img.convert("HSV")
    h, s, v = hsv_img.split()

    # Compute circular hue shift toward base hue
    base_h = base_profile["hue"]
    hue_diff = base_h
    # We'll shift each pixel's hue by a fraction of the shortest path to base_h
    # Build a lookup table for hue channel (0-255)
    h_lut = []
    for i in range(256):
        ph = i / 255.0
        diff = base_h - ph
        if diff > 0.5:
            diff -= 1.0
        elif diff < -0.5:
            diff += 1.0
        shifted = ph + diff * COLOR_MATCH_HUE_SHIFT
        h_lut.append(int(shifted % 1.0 * 255))
    h_new = h.point(h_lut)

    # Saturation: blend each pixel toward base saturation
    base_s = base_profile["saturation"]
    s_lut = []
    for i in range(256):
        ps = i / 255.0
        blended = ps + (base_s - ps) * COLOR_MATCH_SATURATION
        s_lut.append(int(max(0.0, min(1.0, blended)) * 255))
    s_new = s.point(s_lut)

    # Value/brightness: blend each pixel toward base brightness
    base_v = base_profile["brightness"]
    v_lut = []
    for i in range(256):
        pv = i / 255.0
        blended = pv + (base_v - pv) * COLOR_MATCH_BRIGHTNESS
        v_lut.append(int(max(0.0, min(1.0, blended)) * 255))
    v_new = v.point(v_lut)

    hsv_matched = Image.merge("HSV", (h_new, s_new, v_new))
    rgb_matched = hsv_matched.convert("RGB")

    # Set alpha to 0.9 and apply Gaussian blur
    a = a.point(lambda x: int(x * 0.95))
    a = a.filter(ImageFilter.GaussianBlur(radius=2))

    return Image.merge("RGBA", (*rgb_matched.split(), a))

def load_or_compute(json_path, compute_fn):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    data = compute_fn()
    if data:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    return data


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
    if not points:
        return []
    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    projections = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        proj = dx * ux + dy * uy
        projections.append((proj, x, y, dx, dy))
    if not projections:
        return [(x - shifted_x, y - shifted_y) for x, y in points]
    min_proj = min(p[0] for p in projections)
    max_proj = max(p[0] for p in projections)
    cut_proj = (min_proj + max_proj) / 2
    adjusted = []
    for proj, x, y, dx, dy in projections:
        if proj > cut_proj and (dx != 0 or dy != 0):
            boundary = get_rect_boundary(cx, cy, dx, dy, shifted_x, shifted_y, rw, rh)
            if boundary:
                new_x, new_y = boundary
            else:
                new_x, new_y = x, y
        else:
            new_x, new_y = x, y
        adjusted.append((new_x - shifted_x, new_y - shifted_y))
    return adjusted

def get_nail_size(a: float, w: float, h: float):
    # Handle undefined angles
    if abs(a) > 180:
        print(f"Angle {a} falls into undefined cases!")
        return 0.0, 0.0

    # Convert angle from degrees to radians
    rad = math.radians(a)

    # Calculate squared weights
    cos_sq = math.cos(rad) ** 2
    sin_sq = math.sin(rad) ** 2

    # Smoothly blend between H (near 0°, ±180°) and W (near ±90°)
    return (cos_sq * h) + (sin_sq * w), (sin_sq * h) + (cos_sq * w)

def draw_nail_polish(base_image, sample_image, points, cx, cy, angle, w, h, z, a3d, base_color_profile=None):
    sample_img_w, sample_img_h = sample_image.size
    # Skew the nail shape based on the a3d angle.
    # a3d close to 90: make the top of sample_image same width, bottom of sample_image narrow.
    # a3d close to -90: make the bottom of sample_image same width, top of sample_image narrow.
    a3d_val = float(a3d)
    a3d_abs = abs(a3d_val)

    img_np = np.array(sample_image)

    a3d_norm = max(-1.0, min(1.0, a3d_val / 90.0))
    offset = sample_img_w * abs(a3d_norm) * 0.25

    max_x = sample_img_w
    max_y = sample_img_h
    src = np.float32([
        [0, 0],
        [max_x, 0],
        [max_x, max_y],
        [0, max_y]
    ])

    dst = src.copy()
    if a3d_norm < 0:
        dst[0, 0] = offset
        dst[1, 0] = max_x - offset
    else:
        dst[2, 0] = max_x - offset
        dst[3, 0] = offset

    M = cv2.getPerspectiveTransform(src, dst)
    img_np = cv2.warpPerspective(img_np, M, (sample_img_w, sample_img_h), flags=cv2.INTER_LINEAR)
    sample_image = Image.fromarray(img_np)

    ref_w, ref_h = get_nail_size(angle, w, h)
    if a3d_norm > 0:
      ref_w = ref_w * (1.0 + a3d_norm * 0.25)
    ref_height = max((ref_w * sample_img_h / sample_img_w) * math.cos(math.radians(a3d_abs)), ref_h * 1.1)
    resized_img = sample_image.resize((int(ref_w), int(ref_height)), Image.Resampling.LANCZOS)
    rotated_img = resized_img.rotate(-float(angle + 90), expand=True)

    # Apply depth-based brightness
    z = float(z)
    Z_MIN, Z_MAX = -0.08, 0.08
    depth_ratio = (z - Z_MIN) / (Z_MAX - Z_MIN)
    depth_ratio = max(0.0, min(1.0, depth_ratio))
    brightness_factor = 1.0 / (1.0 + depth_ratio * 0.03)
    enhancer = ImageEnhance.Brightness(rotated_img)
    rotated_img = enhancer.enhance(brightness_factor)

    # Apply base image color matching
    if base_color_profile is not None:
        rotated_img = apply_color_matching(rotated_img, base_color_profile)

    rw, rh = rotated_img.size

    paste_x = int(cx - rw / 2)
    paste_y = int(cy - rh / 2)

    shift_center_nail_distance = (h - ref_height) / 2.8

    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)

    shifted_x = int(paste_x - shift_center_nail_distance * ux)
    shifted_y = int(paste_y - shift_center_nail_distance * uy)

    nail_alpha = rotated_img.split()[-1]

    # Generate smooth polygon mask via supersampling to avoid jagged edges.
    supersample = 4
    mask_w, mask_h = rotated_img.size
    big_w, big_h = mask_w * supersample, mask_h * supersample
    big_mask = Image.new("L", (big_w, big_h), 0)
    big_draw = ImageDraw.Draw(big_mask)
    adjusted_points = compute_adjusted_points(points, cx, cy, angle, shifted_x, shifted_y, rw, rh)
    scaled_points = [(x * supersample, y * supersample) for x, y in adjusted_points]
    big_draw.polygon(scaled_points, fill=255)
    big_mask = big_mask.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR * supersample))
    polygon_mask = big_mask.resize((mask_w, mask_h), Image.Resampling.LANCZOS)

    final_mask = ImageChops.multiply(polygon_mask, nail_alpha)
    base_image.paste(rotated_img, (shifted_x, shifted_y), final_mask)

def draw_nail_debug(mask_draw, points, cx, cy, angle, w, h, z, a3d):
    debug_color = (0, 255, 0)
    mask_draw.polygon(points, fill=(255, 255, 255))

    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    px = -uy
    py = ux

    length = math.sqrt(float(w) ** 2 + float(h) ** 2)

    min_proj = float("inf")
    bottom_x, bottom_y = points[0]
    for pt_x, pt_y in points:
        proj = pt_x * ux + pt_y * uy
        if proj < min_proj:
            min_proj = proj
            bottom_x, bottom_y = pt_x, pt_y

    end_x = bottom_x + length * ux
    end_y = bottom_y + length * uy

    mask_draw.line([(cx, cy), (end_x, end_y)], fill=debug_color, width=4)
    head_size = max(8, length * 0.1)
    left_x = end_x - head_size * math.cos(angle_rad - math.radians(30))
    left_y = end_y - head_size * math.sin(angle_rad - math.radians(30))
    right_x = end_x - head_size * math.cos(angle_rad + math.radians(30))
    right_y = end_y - head_size * math.sin(angle_rad + math.radians(30))
    mask_draw.polygon([(end_x, end_y), (left_x, left_y), (right_x, right_y)], fill=debug_color)

    t = (bottom_x - cx) * ux + (bottom_y - cy) * uy
    mid_x = cx + t * ux
    mid_y = cy + t * uy

    mask_draw.ellipse(
        [mid_x - 8, mid_y - 8, mid_x + 8, mid_y + 8],
        fill=(255, 0, 0),
    )

    line_x1 = mid_x - px * w
    line_y1 = mid_y - py * w
    line_x2 = mid_x + px * w
    line_y2 = mid_y + py * w
    mask_draw.line([(line_x1, line_y1), (line_x2, line_y2)], fill=debug_color, width=4)

    nw, nh = get_nail_size(angle, w, h)
    lh = length * 0.2
    tt = cy + length
    f = get_debug_font(length)
    mask_draw.text((cx, tt + lh), f"w: {int(nw)}", fill=debug_color, font=f)
    mask_draw.text((cx, tt + 2 * lh), f"r: {int(w)}x{int(h)}", fill=debug_color, font=f)
    mask_draw.text((cx, tt + 3 * lh), f"a: {int(angle)}°", fill=debug_color, font=f)
    mask_draw.text((cx, tt + 4 * lh), f"z: {z:2f}", fill=debug_color, font=f)
    mask_draw.text((cx, tt + 5 * lh), f"a3d: {int(a3d)}°", fill=debug_color, font=f)
    

def save_and_show_results(mask_image, base_image, output_path=None):
    if output_path is None:
        output_path = "sample-images/static-nail-painting.jpg"
    base_image.save(output_path)
    mask_path = output_path.replace(os.path.splitext(output_path)[1], "-mask" + os.path.splitext(output_path)[1])
    mask_image.save(mask_path)
    # mask_image.show()
    # base_image.show()
    print(f"Saved: {output_path}")
    print(f"Saved mask: {mask_path}")


def main(original_image, reference_image, output_path=None):
    base_image = ImageOps.exif_transpose(Image.open(original_image)).convert("RGB")
    ref_image = Image.open(reference_image).convert("RGBA")
    with open(original_image, "rb") as f:
        image_bytes = f.read()

    width, height = base_image.size

    hands_data = load_or_compute(
        original_image + ".hands_data.json",
        lambda: detect_hands(image_bytes, max_dim=MAX_DETECTION_DIM),
    )
    if not hands_data:
        print(f"No hands detected in {original_image}. Skipping.")
        return False

    filtered_nails = load_or_compute(
        original_image + ".nails_data.json",
        lambda: filter_nails_by_hands(
            detect_nails(image_bytes, max_dim=MAX_DETECTION_DIM),
            hands_data,
            width,
            height,
        ),
    )
    if not filtered_nails:
        print(f"No nails detected in {original_image}. Skipping.")
        return False

    mask_image = Image.new("RGB", (width, height), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    # Analyze base image colors once
    base_color_profile = get_base_image_color_profile(base_image)

    for nail in filtered_nails:
        points = [(float(p["x"]), float(p["y"])) for p in nail.get("points", [])]
        cx = nail.get("x", 0)
        cy = nail.get("y", 0)
        angle = nail.get("angle", 0)
        a3d = nail.get("a3d", 0)
        w = nail.get("width", 0)
        h = nail.get("height", 0)
        z = nail.get("z", 0)

        '''
        MediaPipe defines z as relative depth to the wrist (landmark 0):
            z = 0 at the wrist
            z < 0 → landmark is closer to the camera than the wrist
            z > 0 → landmark is further from the camera than the wrist
        '''
        draw_nail_polish(base_image, ref_image, points, cx, cy, angle, w, h, z, a3d, base_color_profile)
        draw_nail_debug(mask_draw, points, cx, cy, angle, w, h, z, a3d)

    save_and_show_results(mask_image, base_image, output_path)
    return True

if __name__ == "__main__":
    sample_dir = "sample-images"
    ref_image = "sample-images/sample-2.png"
    for filename in sorted(os.listdir(sample_dir)):
        if filename.startswith("hand"):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"} or "-output" in filename:
                continue
            image_path = os.path.join(sample_dir, filename)
            output_path = os.path.join(sample_dir, f"{name}-output{ext}")
            print(f"Processing {image_path} -> {output_path}")
            main(image_path, ref_image, output_path)
