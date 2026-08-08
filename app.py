import base64
import json
import urllib.parse
import urllib.request
import os
from PIL import Image, ImageDraw, ImageOps
import mediapipe as mp
import numpy as np
import cv2

# Default fill color: solid red.
RED = (255, 0, 0)
NAIL_ALPHA = 0.6
NAIL_BLUR = 8
TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)
OUTPUT_IMAGE_PATH = "nails_painted.JPG"

# Configuration
IMAGE_PATH = "IMG_3051.JPG"
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": "wraXc2yVTPswR5wjraOf", "confidence": "0.01"}

# TURNING OFF NAIL FILTERING FOR NOW
SPACE_DETECTION_THRESHOLD = 0.1  # 10% of average image dimension

def detect_nails(image_path=IMAGE_PATH):
    """Send ``image_path`` to the RoBoFlow API and return the inference result."""
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read())

    query_string = urllib.parse.urlencode(PARAMS)
    full_url = f"{URL}?{query_string}"

    req = urllib.request.Request(
        full_url,
        data=base64_encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def _apply_color_transfer(
    bgr: np.ndarray,
    mask: np.ndarray,
    target_hsv: np.ndarray = TARGET_HSV,
    alpha: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = target_hsv[1]

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (bgr.astype(np.float32) * (1.0 - mask_float * alpha) + recolored_bgr.astype(np.float32) * mask_float * alpha).astype(np.uint8)
    return result

def paint_nails(image_path, result, output_path=None, color=RED, alpha: float = 1.0, blur: int = 0):
    """Paint detected nail regions on the image with ``color``.

    Each prediction is expected to carry a ``points`` list describing the
    nail polygon (``mask_format == "polygon"``), i.e. a list of
    ``{"x": float, "y": float}`` vertices.

    Args:
        image_path: Path to the source image file.
        result: Inference result dict returned by the RoBoFlow API.
            Expected to contain a ``predictions`` list whose entries each
            have a ``points`` list of polygon vertices.
        output_path: Destination path for the painted image. When
            ``None`` (the default) ``<image_path>_painted.<ext>`` is used.
        color: RGB color tuple used to fill each nail region. Defaults
            to red ``(255, 0, 0)``.
        alpha: Opacity of the applied color, between 0.0 and 1.0.
            Defaults to 1.0 (fully opaque).
        blur: Gaussian blur radius in pixels applied to the nail mask
            edges to soften them. Set to 0 for no blur.

    Returns:
        The painted ``PIL.Image.Image`` instance (RGB mode), whose
        ``filename`` attribute holds the path the image was saved to.
    """
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    predictions = (result or {}).get("predictions", [])
    
    if predictions:
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        mask = Image.new("L", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask)

        for pred in predictions:
            points = pred.get("points")
            if not points:
                continue
            polygon = [(float(p["x"]), float(p["y"])) for p in points]
            draw_mask.polygon(polygon, fill=255)
        
        mask_np = np.array(mask)
        
        if blur > 0:
            ksize = 2 * int(blur) + 1
            mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur)
        
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        rgb_color = np.uint8([[color]])
        target_hsv = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)

        painted_bgr = _apply_color_transfer(image_bgr, mask_np, target_hsv, alpha=alpha)
        painted_rgb = cv2.cvtColor(painted_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(painted_rgb)
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_painted{ext}"

    image.save(output_path)
    image.filename = output_path
    
    return image

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.5
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]

def detect_hands(image_path):
    """Detect hands in an image and return finger tips in JSON format.

    Args:
        image_path: Path to the image file to process.

    Returns:
        A JSON string representing a list of detected hands. Each hand
        entry contains ``hand_index``, ``handedness``, and a list of
        ``fingertips`` with normalized coordinates.
    """
    image = ImageOps.exif_transpose(Image.open(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)

    results = hands_detector.process(image_np)

    output = []
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_entry = {
                "hand_index": idx,
                "handedness": None,
                "fingertips": []
            }

            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_entry["handedness"] = results.multi_handedness[idx].classification[0].label

            for lm_id in FINGERTIP_IDS:
                lm = hand_landmarks.landmark[lm_id]
                hand_entry["fingertips"].append({
                    "landmark_id": lm_id,
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6)
                })

            output.append(hand_entry)

    return json.dumps(output, indent=2)

def point_in_polygon(x, y, polygon, width, height):
    """Check if point (x, y) is close to the center of polygon.

    The polygon center is computed as the arithmetic mean of its vertices.
    A point is considered inside when its Euclidean distance to the center
    is less than 1%% of the average image dimension.

    Args:
        x: Point x coordinate.
        y: Point y coordinate.
        polygon: List of ``(x, y)`` tuples describing the polygon.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        bool: ``True`` if the point is close to the polygon center.
    """
    if not polygon:
        return False
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    threshold = SPACE_DETECTION_THRESHOLD * (width + height) / 2
    return distance < threshold


def filter_nails_by_hands(nails_result, hands_data, width, height):
    """Filter nail predictions to only those containing at least one fingertip.

    Args:
        nails_result: Inference result dict from RoBoFlow API.
        hands_data: Parsed list of hand entries from ``detect_hands``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A new result dict with ``predictions`` filtered to only nails that
        contain at least one fingertip.
    """
    if not hands_data:
        return {"predictions": []}

    # Flatten all fingertip pixel coordinates
    fingertips_px = []
    for hand in hands_data:
        for tip in hand.get("fingertips", []):
            fingertips_px.append((tip["x"] * width, tip["y"] * height))

    #print("fingertips_px", fingertips_px)

    filtered = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]

        contained = False
        for fx, fy in fingertips_px:
            if point_in_polygon(fx, fy, polygon, width, height):
                contained = True
                break

        if contained:
            filtered.append(pred)

    return {"predictions": filtered}


def main():
    try:
        hands = detect_hands(IMAGE_PATH)
        hands_data = json.loads(hands)

        if hands_data and len(hands_data) > 0:
            # print(f"Detected {len(hands_data)} hands.")
            nails = detect_nails()

            with ImageOps.exif_transpose(Image.open(IMAGE_PATH)) as img:
                width, height = img.size

            filtered_nails = filter_nails_by_hands(nails, hands_data, width, height)

            #print("filtered_nails", filtered_nails)

            predictions = filtered_nails.get("predictions", [])
            if predictions:
                paint_nails(IMAGE_PATH, filtered_nails, output_path=OUTPUT_IMAGE_PATH, alpha=NAIL_ALPHA, blur=NAIL_BLUR)
            else:
                print("No nails detected that contain fingertips. Skipping nail painting.")
        else:
            print("No hands detected. Skipping nail painting.")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")

if __name__ == "__main__":
    main()


