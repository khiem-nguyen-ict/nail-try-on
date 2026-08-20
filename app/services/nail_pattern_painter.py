"""Nail pattern painting service.

Paints a reference pattern image (e.g. a nail-art swatch) onto a single detected
nail region on a base image. The approach is perspective-aware: the pattern is
skewed to match the 3D finger angle, resized/rotated to the nail geometry,
depth-shaded, color-matched to the base image, and composited with an
antialiased polygon mask.

This is intentionally separate from :mod:`app.services.nail_painter`, which
uses an HSV color-transfer + distance-transform gloss technique. Use this
module when you want to stamp a concrete pattern/texture onto the nails rather
than recoloring them.
"""

import math

import cv2
import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageStat,
)

from app.utils.image import apply_color_matching
from app.utils.polygon import compute_adjusted_points, get_nail_size

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# MediaPipe reports ``z`` relative to the wrist (z == 0 at the wrist).
# These bounds define the range mapped to the depth-based brightness ramp.
DEPTH_Z_MIN = -0.08
DEPTH_Z_MAX = 0.08
# Strength of the depth brightness modulation (fractional).
DEPTH_BRIGHTNESS_STRENGTH = 0.03

# Supersampling factor used when rasterizing the polygon mask so that edges
# stay smooth after the subsequent Gaussian blur + downscale.
MASK_SUPERSAMPLE = 4
# Gaussian blur radius (applied at supersampled resolution) for mask feathering.
MASK_BLUR_RADIUS = 6

# How far (in nail-height units) to slide the pattern along the finger axis so
# the painted pattern aligns with the nail's long axis rather than its center.
NAIL_AXIS_SHIFT_DIVISOR = 2.8

# Perspective skew intensity driven by the 3D fingertip angle (``a3d``).
# ``a3d`` of +/-90 maps to an offset of ``A3D_SKEW_FACTOR * image_width``.
A3D_SKEW_FACTOR = 0.25
# Extra width expansion applied to the nail when the finger angles toward the
# camera (``a3d`` positive / finger pointing up toward the viewer).
A3D_WIDTH_EXPANSION = 0.25

# Safety margin multiplier applied to the nail height so the pattern always
# fully covers the nail region.
NAIL_HEIGHT_PAD_RATIO = 1.1

# Strength of the region-based brightness modulation (fractional, 0.0 = off).
REGION_BRIGHTNESS_STRENGTH = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def skew_sample_to_a3d(sample_image, a3d):
    """Horizontally skew a sample image based on the 3D fingertip angle.

    A finger pointing toward the camera (``a3d`` near +90) converges the top
    edge inward; a finger receding (``a3d`` near -90) converges the bottom edge.
    A value near 0 leaves the image untouched.
    """
    a3d_norm = max(-1.0, min(1.0, float(a3d) / 90.0))
    if a3d_norm == 0.0:
        return sample_image

    img_w, img_h = sample_image.size
    offset = img_w * abs(a3d_norm) * A3D_SKEW_FACTOR

    src = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    dst = src.copy()
    if a3d_norm < 0:
        dst[0, 0] = offset
        dst[1, 0] = img_w - offset
    else:
        dst[2, 0] = img_w - offset
        dst[3, 0] = offset

    matrix = cv2.getPerspectiveTransform(src, dst)
    skewed = cv2.warpPerspective(
        np.array(sample_image), matrix, (img_w, img_h), flags=cv2.INTER_LINEAR
    )
    return Image.fromarray(skewed)


def apply_depth_brightness(image, z):
    """Brighten or dim an image based on fingertip depth.

    Nails closer to the camera (lower ``z``) are nudged brighter and nails
    further away are dimmed, relative to the wrist baseline (``z == 0``).
    """
    depth_ratio = (float(z) - DEPTH_Z_MIN) / (DEPTH_Z_MAX - DEPTH_Z_MIN)
    depth_ratio = max(0.0, min(1.0, depth_ratio))
    brightness_factor = 1.0 / (1.0 + depth_ratio * DEPTH_BRIGHTNESS_STRENGTH)
    return ImageEnhance.Brightness(image).enhance(brightness_factor)


def apply_region_brightness(image, base_image, points):
    """Adjust image brightness to match the average luminance of the base region.

    The base image region defined by ``points`` is masked, and its average
    perceptual brightness is computed. The sample ``image`` is then nudged
    toward that brightness by ``REGION_BRIGHTNESS_STRENGTH``.

    Args:
        image: PIL image (RGBA or RGB) of the nail pattern to adjust.
        base_image: PIL image (RGB) of the hand/photo containing the nail.
        points: polygon vertices in base-image pixel space.

    Returns:
        Brightness-adjusted image.
    """
    if not points:
        return image

    mask = Image.new("L", base_image.size, 0)
    ImageDraw.Draw(mask).polygon(
        [(float(x), float(y)) for x, y in points], fill=255
    )

    bbox = mask.getbbox()
    if bbox is None:
        return image

    cropped_base = base_image.crop(bbox).convert("RGB")
    cropped_mask = mask.crop(bbox)

    stat = ImageStat.Stat(cropped_base, cropped_mask)
    r, g, b = stat.mean
    region_brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    sample_rgb = image.convert("RGB")
    sample_stat = ImageStat.Stat(sample_rgb)
    sr, sg, sb = sample_stat.mean
    sample_brightness = (0.299 * sr + 0.587 * sg + 0.114 * sb) / 255.0

    if sample_brightness < 1e-6:
        return image

    target_factor = region_brightness / sample_brightness
    target_factor = max(0.5, min(1.5, target_factor))

    factor = 0.7 + (target_factor - 1.0) * REGION_BRIGHTNESS_STRENGTH
    return ImageEnhance.Brightness(image).enhance(factor)


def build_smooth_mask(points, rw, rh, pattern_alpha):
    """Build an antialiased, feathered polygon mask clipped to the pattern alpha.

    Args:
        points: polygon vertices in the rotated pattern's local coordinate space.
        rw, rh: size of the rotated pattern image.
        pattern_alpha: single-channel alpha of the rotated pattern (used as a
            second clip so transparent padding in the pattern is not pasted).
    """
    supersample = MASK_SUPERSAMPLE
    big_mask = Image.new("L", (rw * supersample, rh * supersample), 0)
    ImageDraw.Draw(big_mask).polygon(
        [(x * supersample, y * supersample) for x, y in points],
        fill=255,
    )
    big_mask = big_mask.filter(
        ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS * supersample)
    )
    polygon_mask = big_mask.resize((rw, rh), Image.Resampling.LANCZOS)
    return ImageChops.multiply(polygon_mask, pattern_alpha)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def paint_nail_pattern(
    base_image,
    sample_image,
    points,
    cx,
    cy,
    angle,
    w,
    h,
    z,
    a3d,
    base_color_profile=None,
):
    """Paint a reference *pattern* onto a single nail region of *base_image*.

    The pattern is pasted in place; ``base_image`` is mutated (consistent with
    the original experiment, where successive nails are layered onto the same
    canvas).

    Args:
        base_image: PIL image (RGB) of the hand/photo to paint onto. Mutated.
        sample_image: PIL image (typically RGBA) holding the nail-art pattern.
        points: list of ``(x, y)`` polygon vertices for the nail region
            (in base-image pixel space).
        cx, cy: center of the nail region (pixel space).
        angle: nail orientation in degrees (long-axis angle from the fingertip
            detector).
        w, h: nail width/height used to size the pattern.
        z: MediaPipe fingertip ``z`` (depth) for brightness shading.
        a3d: 3D fingertip angle (degrees) used for perspective skew.
        base_color_profile: optional HSV profile (from
            :func:`app.utils.image.get_base_image_color_profile`) used to
            harmonize the pattern's colors with the base image.

    Returns:
        The (mutated) ``base_image`` for convenience.
    """
    sample_img_w, sample_img_h = sample_image.size
    sample_ratio = sample_img_h / sample_img_w

    a3d_norm = max(-1.0, min(1.0, float(a3d) / 90.0))

    # 1. Perspective skew driven by the 3D finger angle.
    skewed = skew_sample_to_a3d(sample_image, a3d)

    # 2. Resize to the nail's footprint.
    ref_w, ref_h = get_nail_size(angle, w, h)
    if a3d_norm > 0:
        ref_w = ref_w * (1.0 + a3d_norm * A3D_WIDTH_EXPANSION)
    a3d_abs = abs(float(a3d))
    ref_height = max(
        (ref_w * sample_ratio) * math.cos(math.radians(a3d_abs)),
        ref_h * NAIL_HEIGHT_PAD_RATIO,
    )
    resized_img = skewed.resize(
        (int(ref_w), int(ref_height)), Image.Resampling.LANCZOS
    )

    # 3. Rotate to align with the nail's long axis.
    rotated_img = resized_img.rotate(-float(angle + 90), expand=True)

    # 4. Depth-based brightness + optional color matching.
    rotated_img = apply_depth_brightness(rotated_img, z)
    if base_color_profile is not None:
        rotated_img = apply_color_matching(rotated_img, base_color_profile)

    # 4.5 Region-based brightness adjustment using the original nail region.
    rotated_img = apply_region_brightness(rotated_img, base_image, points)

    rw, rh = rotated_img.size

    # 5. Position and shift the pattern along the finger axis.
    paste_x = int(cx - rw / 2)
    paste_y = int(cy - rh / 2)
    shift_center_nail_distance = (h - ref_height) / NAIL_AXIS_SHIFT_DIVISOR
    angle_rad = math.radians(float(angle))
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    shifted_x = int(paste_x - shift_center_nail_distance * ux)
    shifted_y = int(paste_y - shift_center_nail_distance * uy)

    # 6. Antialiased polygon mask clipped to the pattern alpha.
    nail_alpha = rotated_img.split()[-1]
    mask_points = compute_adjusted_points(
        points, cx, cy, angle, shifted_x, shifted_y, rw, rh
    )
    final_mask = build_smooth_mask(mask_points, rw, rh, nail_alpha)

    # 7. Composite onto the base image.
    base_image.paste(rotated_img, (shifted_x, shifted_y), final_mask)
    return base_image
