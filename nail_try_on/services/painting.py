import os
from io import BytesIO
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from nail_try_on.config import MAX_DETECTION_DIM, NAIL_ALPHA, NAIL_BLUR, RED, ROBOFLOW_MAX_DIM, SPACE_DETECTION_THRESHOLD, TARGET_HSV
from nail_try_on.services.hands import _detect_hands


def _apply_color_transfer(
    bgr: np.ndarray,
    mask: np.ndarray,
    target_hsv: np.ndarray = TARGET_HSV,
    alpha: float = 1.0,
) -> np.ndarray:
    """Replace colors inside ``mask`` with ``target_hsv`` while preserving luminance."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = target_hsv[1]
    hsv[:, :, 2] = target_hsv[2]

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (
        bgr.astype(np.float32) * (1.0 - mask_float * alpha)
        + recolored_bgr.astype(np.float32) * mask_float * alpha
    ).astype(np.uint8)
    return result


def _paint_nails(
    image_source: Union[str, bytes],
    regions,
    color=(255, 0, 0),
    alpha: float = NAIL_ALPHA,
    blur: int = NAIL_BLUR,
    gloss_intensity: float = 0.5,
    preloaded_image: Optional[Image.Image] = None,
):
    """Paint detected nail regions with realistic glossy effect (Optimized)."""
    if preloaded_image is not None:
        image = preloaded_image.convert("RGB")
    elif isinstance(image_source, bytes):
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_source))).convert("RGB")
    else:
        image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")

    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # 1. Create masks for nails and gloss effect
    mask_np = np.zeros((h, w), dtype=np.uint8)
    gloss_mask_np = np.zeros((h, w), dtype=np.float32)

    for pred in regions:
        points = pred.get("points")
        if not points:
            continue

        # Convert points to numpy array of shape (N, 2) with integer coordinates
        poly_np = np.array(
            [(float(p["x"]), float(p["y"])) for p in points], dtype=np.int32
        )

        # draw the polygon on the mask
        cv2.fillPoly(mask_np, [poly_np], 255)

        # Glossy effect: Create a distance transform to simulate light reflection
        if gloss_intensity > 0:
            # Get bounding box of the polygon to limit the distance transform computation
            x, y, bw, bh = cv2.boundingRect(poly_np)
            if bw <= 0 or bh <= 0:
                continue

            # Cut out the region of interest for distance transform
            crop_mask = np.zeros((bh, bw), dtype=np.uint8)
            crop_poly = poly_np - [x, y]
            cv2.fillPoly(crop_mask, [crop_poly], 255)

            # Distance transform to create a gradient for the gloss effect
            dist = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
            max_val = dist.max()

            if max_val > 0:
                # Increase the highlight effect by raising to a power (1.8) for a more pronounced glossy look
                highlight = (dist / max_val) ** 1.8

                # Update the gloss mask in the original image space
                crop_gloss = gloss_mask_np[y : y + bh, x : x + bw]
                np.maximum(crop_gloss, highlight, out=crop_gloss)

    # 2. Blur the masks to create a smooth transition
    if blur > 0:
        ksize = 2 * int(blur) + 1
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur)
        if gloss_intensity > 0:
            gloss_mask_np = cv2.GaussianBlur(gloss_mask_np, (ksize, ksize), sigmaX=blur)

    # 3. Apply color transfer to the nail regions
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    rgb_color = np.uint8([[color]])
    target_hsv = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)

    painted_bgr = _apply_color_transfer(image_bgr, mask_np, target_hsv, alpha=alpha)

    # 4. Glossy effect: Adjust brightness and saturation based on the gloss mask
    if gloss_intensity > 0:
        painted_hsv = cv2.cvtColor(painted_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Improve brightness channel V based on the gloss mask
        v_channel = painted_hsv[:, :, 2]
        v_channel += gloss_mask_np * (gloss_intensity * 110.0)
        np.clip(v_channel, 0, 255, out=v_channel)

        # Describe saturation channel S based on the gloss mask to reduce saturation in glossy areas
        s_channel = painted_hsv[:, :, 1]
        s_channel *= 1.0 - (gloss_mask_np * gloss_intensity * 0.35)
        np.clip(s_channel, 0, 255, out=s_channel)

        painted_hsv[:, :, 1] = s_channel
        painted_hsv[:, :, 2] = v_channel

        painted_bgr = cv2.cvtColor(painted_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Output the final image as bytes or PIL Image
    painted_rgb = cv2.cvtColor(painted_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(painted_rgb)

    if isinstance(image_source, bytes):
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    return image


def _point_in_polygon(x, y, polygon, width, height):
    """Check if point (x, y) is close to the center of polygon."""
    if not polygon:
        return False
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    threshold = SPACE_DETECTION_THRESHOLD * (width + height) / 2
    return distance < threshold


def _filter_nails_by_hands(nails_result, hands_data, width, height):
    """Filter nail predictions to only those containing at least one fingertip."""
    if not hands_data:
        return []

    fingertips_px = []
    for hand in hands_data:
        for tip in hand.get("fingertips", []):
            fingertips_px.append((tip["x"] * width, tip["y"] * height, tip["angle"]))

    filtered = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]

        contained = False
        for fx, fy, angle in fingertips_px:
            if _point_in_polygon(fx, fy, polygon, width, height):
                pred["fingertip_angle"] = angle  # Store the fingertip angle in the prediction
                contained = True
                break

        if contained:
            filtered.append(pred)

    return filtered


def _create_mask_from_polygon(width, height, polygon):
    """Create a binary mask from polygon predictions."""
    try:
        all_polygons = []
        for prediction in polygon:
            if prediction["class"] == "Nail" and "points" in prediction:
                poly_points = [[pt["x"], pt["y"]] for pt in prediction["points"]]
                all_polygons.append(np.array(poly_points, dtype=np.int32))

        # Create mask with exact dimensions
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, all_polygons, 255)
    except Exception as e:
        print("Error creating mask from polygon:", str(e))
        mask = np.zeros((height, width), dtype=np.uint8)
    return mask

def process_frame_with_hand_status(
    image_bytes: bytes,
    max_dim: int = MAX_DETECTION_DIM,
    roboflow_max_dim: int = ROBOFLOW_MAX_DIM,
    color: tuple = RED,
    alpha: float = NAIL_ALPHA,
    blur: int = NAIL_BLUR,
    live_preview: bool = True,
) -> tuple[bytes, bool]:
    """Process a frame and return (processed_bytes, hands_found_bool)."""
    from nail_try_on.services.roboflow import detect_nails

    try:
        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            image = img.convert("RGB")
            width, height = image.size

        # Detect hands. If hands are found, proceed to detect nails and paint them.
        # Otherwise, return the original image.
        hands_data = _detect_hands(image_bytes, max_dim=max_dim, preloaded_image=image)

        if hands_data and len(hands_data) > 0:
            print(f"Detected {len(hands_data)} hands.")
            # Detect nails and filter by hands.
            nails = detect_nails(image_bytes, max_dim=roboflow_max_dim)

            nails_data = _filter_nails_by_hands(nails, hands_data, width, height)
            if nails_data and len(nails_data) > 0:
                print(f"Detected {len(nails_data)} nail regions after filtering by hands.")
                result = _paint_nails(
                    image_bytes,
                    nails_data,
                    color=color,
                    alpha=alpha,
                    blur=blur,
                    preloaded_image=image,
                )
                return result, True
    except Exception as e:
        print("Error processing frame:", str(e))
    return image_bytes, False
