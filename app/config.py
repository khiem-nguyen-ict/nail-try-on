import os
import warnings

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Default fill color: solid red.
RED = (255, 0, 0)
NAIL_ALPHA = float(os.getenv("NAIL_ALPHA", "0.4"))
MAX_PROCESS_FPS = int(os.getenv("MAX_PROCESS_FPS", "20"))
NO_HAND_COOLDOWN = float(os.getenv("NO_HAND_COOLDOWN", "1.0"))
FRAME_SKIPPED_BLUR_THRESHOLD = float(os.getenv("FRAME_SKIPPED_BLUR_THRESHOLD", "50.0"))
MAX_CAPTURE_DIM = int(os.getenv("MAX_CAPTURE_DIM", "1280"))
MAX_SEND_FPS = int(os.getenv("MAX_SEND_FPS", "10"))

# Local constants (not configurable via env)
NAIL_BLUR = 1
YOLO_CONFIDENCE_THRESHOLD = 0.5

TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)

# Configuration
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": os.getenv("ROBOFLOW_API_KEY", ""), "confidence": YOLO_CONFIDENCE_THRESHOLD}

if not os.getenv("ROBOFLOW_API_KEY"):
    warnings.warn(
        "ROBOFLOW_API_KEY environment variable is not set. Nail detection features will not work. "
        "Set it in your deployment environment (e.g., Render dashboard) to enable them.",
        RuntimeWarning,
    )
