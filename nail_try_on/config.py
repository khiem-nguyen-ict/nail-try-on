import os
import warnings
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Suppress protobuf deprecation warning from MediaPipe
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


def _get_env_int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def _get_env_float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string (e.g. '#FF0000' or 'FF0000') to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return RED
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return RED


# Default fill color: solid red.
RED = (255, 0, 0)

# Paint settings
NAIL_ALPHA = _get_env_float("NAIL_ALPHA", "0.4")
NAIL_BLUR = _get_env_int("NAIL_BLUR", "1")
SPACE_DETECTION_THRESHOLD = _get_env_float("SPACE_DETECTION_THRESHOLD", "0.1")

# Hand detection settings
YOLO_CONFIDENCE_THRESHOLD = _get_env_float("YOLO_CONFIDENCE_THRESHOLD", "0.5")
MAX_DETECTION_DIM = _get_env_int("MAX_DETECTION_DIM", "320")
MAX_PROCESS_FPS = _get_env_int("MAX_PROCESS_FPS", "20")
NO_HAND_COOLDOWN = _get_env_float("NO_HAND_COOLDOWN", "1.0")

# Roboflow settings
ROBOFLOW_MAX_DIM = _get_env_int("ROBOFLOW_MAX_DIM", "1024")

# Capture / send settings
MAX_CAPTURE_DIM = _get_env_int("MAX_CAPTURE_DIM", "1280")
MAX_SEND_FPS = _get_env_int("MAX_SEND_FPS", "10")

# JPEG compression quality (0-100) for captured frames sent to the server
IMAGE_QUALITY = _get_env_int("IMAGE_QUALITY", "80")

# Color transfer target (HSV space)
TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)

# Roboflow API
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
ROBOFLOW_PARAMS = {
    "api_key": ROBOFLOW_API_KEY,
    "confidence": YOLO_CONFIDENCE_THRESHOLD,
}


def get_static_dir() -> Path:
    """Return the path to the static assets directory."""
    return BASE_DIR / "static"
