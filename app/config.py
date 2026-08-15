import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Default fill color: solid red.
RED = (255, 0, 0)
NAIL_ALPHA = float(os.getenv("NAIL_ALPHA", "0.4"))
NAIL_BLUR = int(os.getenv("NAIL_BLUR", "1"))
SPACE_DETECTION_THRESHOLD = float(os.getenv("SPACE_DETECTION_THRESHOLD", "0.1"))
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
MAX_DETECTION_DIM = int(os.getenv("MAX_DETECTION_DIM", "320"))
MAX_PROCESS_FPS = int(os.getenv("MAX_PROCESS_FPS", "20"))
NO_HAND_COOLDOWN = float(os.getenv("NO_HAND_COOLDOWN", "1.0"))
ROBOFLOW_MAX_DIM = int(os.getenv("ROBOFLOW_MAX_DIM", "1024"))
FRAME_SKIPPED_BLUR_THRESHOLD = float(os.getenv("FRAME_SKIPPED_BLUR_THRESHOLD", "50.0"))
MAX_CAPTURE_DIM = int(os.getenv("MAX_CAPTURE_DIM", "1280"))
MAX_SEND_FPS = int(os.getenv("MAX_SEND_FPS", "10"))
# JPEG compression quality (0-100) for captured frames sent to the server
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "80"))

TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)

# Configuration
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": os.getenv("ROBOFLOW_API_KEY", ""), "confidence": YOLO_CONFIDENCE_THRESHOLD}

if not os.getenv("ROBOFLOW_API_KEY"):
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set. Please set it to your RoBoFlow API key.")
