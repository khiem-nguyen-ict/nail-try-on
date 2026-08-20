import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.image import get_base_image_color_profile
from app.utils.polygon import get_nail_size

from app.services.hand_detector import detect_hands
from app.services.nail_detector import detect_nails, filter_nails_by_hands
from app.services.nail_pattern_painter import paint_nail_pattern
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math

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

def load_or_compute(json_path, compute_fn):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    data = compute_fn()
    if data:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    return data

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
    base_image.save(output_path)
    mask_path = output_path.replace(os.path.splitext(output_path)[1], "-mask" + os.path.splitext(output_path)[1])
    mask_image.save(mask_path)
    # mask_image.show()
    # base_image.show()
    print(f"Saved: {output_path}")
    print(f"Saved mask: {mask_path}")


def main(original_image, reference_image, output_path):
    base_image = ImageOps.exif_transpose(Image.open(original_image)).convert("RGB")
    ref_image = Image.open(reference_image).convert("RGBA")
    with open(original_image, "rb") as f:
        image_bytes = f.read()

    width, height = base_image.size

    f = output_path.replace(os.path.splitext(output_path)[1], "-mp-debug" + os.path.splitext(output_path)[1])

    hands_data = load_or_compute(
        original_image + ".hands_data.json",
        lambda: detect_hands(image_bytes, debug_save_path=f),
    )
    if not hands_data:
        print(f"No hands detected in {original_image}. Skipping.")
        return False

    filtered_nails = load_or_compute(
        original_image + ".nails_data.json",
        lambda: filter_nails_by_hands(
            detect_nails(image_bytes),
            hands_data,
            width,
            height,
        ),
    )
    if not filtered_nails:
        print(f"No nails detected in {original_image}. Skipping.")
        return False

    filtered_nails.sort(key=lambda nail: (sum(p["z"] for p in nail.get("points", [])), nail.get("a3d", 0)))

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
        paint_nail_pattern(base_image, ref_image, points, cx, cy, angle, w, h, z, a3d, base_color_profile)
        draw_nail_debug(mask_draw, points, cx, cy, angle, w, h, z, a3d)

    save_and_show_results(mask_image, base_image, output_path)
    return True

if __name__ == "__main__":
    sample_dir = "sample-images"
    ref_image = "sample-images/sample-2.png"
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
    EXCLUDED_PATTERNS = ("-output", "-enhance", "-debug", "-mp-debug")
    for filename in sorted(os.listdir(sample_dir)):
        if filename.startswith("hand"):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in ALLOWED_EXTENSIONS or any(pattern in filename for pattern in EXCLUDED_PATTERNS):
                continue
            image_path = os.path.join(sample_dir, filename)
            output_path = os.path.join(sample_dir, f"{name}-output{ext}")
            print(f"Processing {image_path} -> {output_path}")
            main(image_path, ref_image, output_path)
