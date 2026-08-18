import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from typing import Union

from app.config import TARGET_HSV


def _apply_color_transfer(
    bgr: np.ndarray,
    mask: np.ndarray,
    target_hsv: np.ndarray = TARGET_HSV,
    alpha: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = target_hsv[1]
    hsv[:, :, 2] = target_hsv[2]

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (bgr.astype(np.float32) * (1.0 - mask_float * alpha) + recolored_bgr.astype(np.float32) * mask_float * alpha).astype(np.uint8)
    return result


def paint_nails(
    image_source: Union[str, bytes], 
    regions, 
    color=(255, 0, 0), 
    alpha: float = 1.0, 
    blur: int = 0, 
    gloss_intensity: float = 0.5, # Độ bóng (0.0: tắt, 0.5: vừa, 1.0: bóng mạnh)
    preloaded_image: Union[Image.Image, None] = None
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

    # 1. CReate masks for nails and gloss effect
    mask_np = np.zeros((h, w), dtype=np.uint8)
    gloss_mask_np = np.zeros((h, w), dtype=np.float32)

    for pred in regions:
        points = pred.get("points")
        if not points:
            continue
        
        # Convert points to numpy array of shape (N, 2) with integer coordinates
        poly_np = np.array([(float(p["x"]), float(p["y"])) for p in points], dtype=np.int32)
        
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
                # Inscrease the highlight effect by raising to a power (1.8) for a more pronounced glossy look
                highlight = (dist / max_val) ** 1.8
                
                # Update the gloss mask in the original image space
                crop_gloss = gloss_mask_np[y:y+bh, x:x+bw]
                np.maximum(crop_gloss, highlight, out=crop_gloss)

    # 2. Blur the masks to create a smooth transition
    if blur > 0:
        ksize = 2 * int(blur) + 1
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur)
        if gloss_intensity > 0:
            gloss_mask_np = cv2.GaussianBlur(gloss_mask_np, (ksize, ksize), sigmaX=blur)

    # VALIDATION: if mask is empty, return original image
    if np.count_nonzero(mask_np) == 0:
        if isinstance(image_source, bytes):
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        return image

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
        s_channel *= (1.0 - (gloss_mask_np * gloss_intensity * 0.35))
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
