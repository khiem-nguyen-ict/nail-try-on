import cv2
import numpy as np
import colorsys
from PIL import Image, ImageFilter, ImageStat

# Color matching defaults
COLOR_MATCH_HUE_SHIFT = 0.04       # max hue shift toward base image (0-1, fraction of hue circle)
COLOR_MATCH_SATURATION = 0.15      # how much to blend saturation toward base image (0-1)
COLOR_MATCH_BRIGHTNESS = 0.2       # how much to blend brightness toward base image (0-1)

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