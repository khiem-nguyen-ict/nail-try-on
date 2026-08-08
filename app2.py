from __future__ import annotations

import os
import base64
import threading
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from ultralytics import YOLO

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH: str = "./models/best_int8.onnx"
TARGET_RGB: Tuple[int, int, int] = (200, 20, 60)
TARGET_HSV: np.ndarray = cv2.cvtColor(
    np.array([TARGET_RGB[2], TARGET_RGB[1], TARGET_RGB[0]], dtype=np.uint8).reshape(1, 1, 3),
    cv2.COLOR_BGR2HSV,
)[0, 0]
YOLO_CONF: float = 0.25
ROI_MAX_WIDTH: int = 320
EMA_ALPHA: float = 0.35
OPTICAL_FLOW_WIN_SIZE: int = 21
OPTICAL_FLOW_MAX_LEVEL: int = 2
MORPH_KERNEL_SIZE: int = 3
GAUSSIAN_KERNEL_SIZE: int = 5
MAX_HANDS: int = 2
FLOW_MAGNITUDE_THRESHOLD: float = 5.0
FLOW_ERROR_THRESHOLD: float = 0.35

# ─────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────
_model = None
_model_ready = False
_model_lock = threading.Lock()

# ─────────────────────────────────────────────
# Global temporal state (thread-safe)
# ─────────────────────────────────────────────
_temporal_lock = threading.Lock()
_prev_gray = None
_prev_mask = None
_ema_mask = None


# ─────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────
def _load_model() -> YOLO:
    global _model
    with _model_lock:
        if _model is None:
            _model = YOLO(MODEL_PATH, task="segment")
    return _model


def _warmup_model() -> None:
    global _model_ready
    if _model_ready:
        return
    m = _load_model()
    test_img = np.zeros((64, 64, 3), dtype=np.uint8)
    test_img[:] = 128
    try:
        m.predict(test_img, conf=YOLO_CONF, verbose=False)
    except Exception:
        pass
    with _model_lock:
        _model_ready = True


# ─────────────────────────────────────────────
# Natural nail contour generator
# ─────────────────────────────────────────────
def _natural_nail_contour(
    cx: float,
    cy: float,
    a: float,
    b: float,
    n_distal: float = 1.5,
    n_proximal: float = 2.5,
    taper: float = -0.15,
    angle_deg: float = 0.0,
    num_points: int = 72,
    shrink_factor: float = 0.82,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    n = np.where(sin_t >= 0.0, n_distal, n_proximal)

    a_s = a * shrink_factor
    b_s = b * shrink_factor

    abs_cos = np.abs(cos_t)
    abs_sin = np.abs(sin_t)
    x_base = np.sign(cos_t) * np.power(abs_cos, 2.0 / n) * a_s
    y_base = np.sign(sin_t) * np.power(abs_sin, 2.0 / n) * b_s

    taper_factor = 1.0 + taper * (y_base / b_s)
    x_tapered = x_base * taper_factor

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    x_rot = x_tapered * cos_a - y_base * sin_a + cx
    y_rot = x_tapered * sin_a + y_base * cos_a + cy

    pts = np.column_stack((
        np.round(x_rot).astype(np.int32),
        np.round(y_rot).astype(np.int32),
    ))
    return pts


# ─────────────────────────────────────────────
# MediaPipe helpers
# ─────────────────────────────────────────────
_mp_hands = None
_mp_hands_lock = threading.Lock()


def _get_mp_hands() -> Optional[Any]:
    global _mp_hands
    if not _MEDIAPIPE_AVAILABLE:
        return None
    if _mp_hands is None:
        with _mp_hands_lock:
            if _mp_hands is None:
                try:
                    mp_hands_mod = __import__("mediapipe", fromlist=["hands"])
                    _mp_hands = mp_hands_mod.solutions.hands.Hands(
                        model_complexity=0,
                        static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                    )
                except Exception:
                    return None
    return _mp_hands


def _process_hand_landmarks(
    landmarks: Any,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, float]:
    tip_indices = [4, 8, 12, 16, 20]
    mcp_indices = [1, 5, 9, 13, 17]
    tips = []
    mcps = []
    for idx in tip_indices:
        lm = landmarks.landmark[idx]
        tips.append((lm.x * img_w, lm.y * img_h, lm.z))
    for idx in mcp_indices:
        lm = landmarks.landmark[idx]
        mcps.append((lm.x * img_w, lm.y * img_h))
    tips_arr = np.array(tips, dtype=np.float32)
    mcps_arr = np.array(mcps, dtype=np.float32)
    diffs = mcps_arr[:, None, :] - mcps_arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    max_dist = float(np.max(dists))
    hand_scale = max((max_dist / 80.0), 0.3)
    return tips_arr, hand_scale


def _detect_all_hands(bgr_small: np.ndarray) -> Tuple[List[np.ndarray], float]:
    hands = _get_mp_hands()
    if hands is None:
        return [], 1.0
    rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    try:
        result = hands.process(rgb_small)
    except Exception:
        return [], 1.0
    if not result.multi_hand_landmarks:
        return [], 1.0

    all_tips = []
    overall_scale = 1.0
    for hl in result.multi_hand_landmarks:
        tips, scale = _process_hand_landmarks(hl, bgr_small.shape[1], bgr_small.shape[0])
        all_tips.append(tips)
        overall_scale = max(overall_scale, scale)
    return all_tips, overall_scale


# ─────────────────────────────────────────────
# ROI mask generation
# ─────────────────────────────────────────────
def _build_fingertip_roi_mask(
    all_fingertips: List[np.ndarray],
    frame_shape: Tuple[int, int],
    hand_scale: float,
) -> np.ndarray:
    h, w = frame_shape
    base_radius = max(int(18.0 * hand_scale), 8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    # Thumb=0, Index=1, Middle=2, Ring=3, Little=4
    radius_multipliers = [1.4, 1.0, 1.0, 1.0, 0.7]
    for tips in all_fingertips:
        for i in range(min(tips.shape[0], len(radius_multipliers))):
            fx = int(np.clip(tips[i, 0], 0, w - 1))
            fy = int(np.clip(tips[i, 1], 0, h - 1))
            radius = max(int(base_radius * radius_multipliers[i]), 6)
            cv2.circle(roi_mask, (fx, fy), radius, 255, -1)
    return roi_mask


# ─────────────────────────────────────────────
# Mask post-processing (OpenCV only)
# ─────────────────────────────────────────────
def _postprocess_mask(raw_mask: np.ndarray) -> np.ndarray:
    k = MORPH_KERNEL_SIZE
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.GaussianBlur(m, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    return m


# ─────────────────────────────────────────────
# Temporal stabilization
# ─────────────────────────────────────────────
def _temporal_stabilize(
    curr_mask: np.ndarray,
    curr_gray: np.ndarray,
    frame_shape: Tuple[int, int],
) -> Tuple[np.ndarray, float, bool]:
    global _prev_gray, _prev_mask, _ema_mask

    h, w = frame_shape
    stabilized = curr_mask.copy()
    mean_error = 0.0
    flow_confident = True

    with _temporal_lock:
        if _ema_mask is not None and _ema_mask.shape != curr_mask.shape:
            _ema_mask = cv2.resize(_ema_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if _prev_mask is not None and _prev_mask.shape != curr_mask.shape:
            _prev_mask = cv2.resize(_prev_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if _prev_gray is not None and _prev_gray.shape != curr_gray.shape:
            _prev_gray = cv2.resize(_prev_gray, (w, h), interpolation=cv2.INTER_LINEAR)

        if _prev_gray is not None and _prev_mask is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    _prev_gray,
                    curr_gray,
                    None,
                    0.5,
                    OPTICAL_FLOW_MAX_LEVEL,
                    OPTICAL_FLOW_WIN_SIZE,
                    3,
                    5,
                    1.2,
                    0,
                )

                # Flow magnitude for confidence assessment
                flow_magnitude = np.sqrt(flow[..., 0].astype(np.float32) ** 2 + flow[..., 1].astype(np.float32) ** 2)
                mean_flow = float(np.mean(flow_magnitude))

                # If flow magnitude is too high, flow vectors are unreliable — skip flow-based stabilization
                if mean_flow > FLOW_MAGNITUDE_THRESHOLD:
                    flow_confident = False
                else:
                    flow_half = cv2.resize(flow, (w // 2, h // 2))
                    h_map = flow_half[..., 0].astype(np.float32)
                    v_map = flow_half[..., 1].astype(np.float32)
                    map_x, map_y = _build_remap_maps(h_map, v_map, w, h)
                    warped_prev = cv2.remap(_prev_mask, map_x, map_y, cv2.INTER_LINEAR)

                    # Mean error: how much the warped previous mask differs from the current mask
                    mean_error = float(
                        np.mean(np.abs(curr_mask.astype(np.float32) - warped_prev.astype(np.float32))) / 255.0
                    )

                    if _ema_mask is None:
                        _ema_mask = warped_prev.astype(np.float32)
                    else:
                        _ema_mask = (
                            EMA_ALPHA * warped_prev.astype(np.float32)
                            + (1.0 - EMA_ALPHA) * _ema_mask
                        )
                    stabilized = cv2.addWeighted(
                        curr_mask, 1.0 - EMA_ALPHA,
                        np.clip(_ema_mask, 0, 255).astype(np.uint8),
                        EMA_ALPHA,
                        0,
                    )
            except Exception:
                pass

        if _ema_mask is None:
            _ema_mask = curr_mask.astype(np.float32)
        else:
            _ema_mask = (
                EMA_ALPHA * curr_mask.astype(np.float32)
                + (1.0 - EMA_ALPHA) * _ema_mask
            )

        _prev_gray = curr_gray.copy()
        _prev_mask = curr_mask.copy()

    return stabilized, mean_error, flow_confident


# ─────────────────────────────────────────────
# Remap map builder
# ─────────────────────────────────────────────
def _build_remap_maps(
    h_flow: np.ndarray,
    v_flow: np.ndarray,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    if w > 1 and h > 1:
        small_x = np.arange(w // 2, dtype=np.float32)
        small_y = np.arange(h // 2, dtype=np.float32)
        x_base, y_base = np.meshgrid(small_x, small_y)
        x_full = cv2.resize(x_base, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        y_full = cv2.resize(y_base, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        h_scaled = cv2.resize(h_flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        v_scaled = cv2.resize(v_flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        map_x[:] = x_full + h_scaled
        map_y[:] = y_full + v_scaled

    return map_x, map_y


# ─────────────────────────────────────────────
# Color transfer (OpenCV HSV)
# ─────────────────────────────────────────────
def _apply_color_transfer(
    bgr: np.ndarray,
    mask: np.ndarray,
    target_hsv: np.ndarray = TARGET_HSV,
) -> np.ndarray:

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = target_hsv[1]

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (bgr.astype(np.float32) * (1.0 - mask_float) + recolored_bgr.astype(np.float32) * mask_float).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# YOLO mask builder
# ─────────────────────────────────────────────
def _build_combined_mask(
    yolo_results: list,
    roi_mask: np.ndarray,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = frame_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if len(yolo_results) == 0 or not hasattr(yolo_results[0], "masks") or yolo_results[0].masks is None:
        return combined_mask

    masks = yolo_results[0].masks

    if masks is not None and masks.xy is not None:
        polys = masks.xy
        for idx, polygon in enumerate(polys):
            if polygon is None or len(polygon) < 5:
                continue
            try:
                contour = polygon.astype(np.float32)
                if len(contour) < 5:
                    continue
                (cx, cy), (axes_x, axes_y), angle = cv2.fitEllipse(contour)

                pts = _natural_nail_contour(
                    cx=cx,
                    cy=cy,
                    a=axes_x / 2.0,
                    b=axes_y / 2.0,
                    n_distal=1.5,
                    n_proximal=2.5,
                    taper=-0.15,
                    angle_deg=angle,
                    shrink_factor=0.82,
                )
                if len(pts) >= 3:
                    cv2.fillPoly(combined_mask, [pts], 255)
            except Exception:
                continue

    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
    return combined_mask


# ─────────────────────────────────────────────
# Parallel inference helpers
# ─────────────────────────────────────────────
def _run_mediapipe(
    small_frame: np.ndarray,
    result_holder: dict,
) -> None:
    try:
        all_tips, hand_scale = _detect_all_hands(small_frame)
        result_holder["tips"] = all_tips
        result_holder["scale"] = hand_scale
    except Exception:
        result_holder["tips"] = []
        result_holder["scale"] = 1.0


def _run_yolo(
    model: YOLO,
    frame: np.ndarray,
    result_holder: dict,
) -> None:
    try:
        results = model.predict(frame, conf=YOLO_CONF, verbose=False)
        result_holder["results"] = results
    except Exception:
        result_holder["results"] = []


# ─────────────────────────────────────────────
# Core processing pipeline
# ─────────────────────────────────────────────
def _process_frame(
    jpeg_bytes: bytes,
) -> bytes:
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG")

    h, w = frame.shape[:2]

    scale = ROI_MAX_WIDTH / float(w)
    small_w = ROI_MAX_WIDTH
    small_h = int(h * scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # ── Parallel: MediaPipe + YOLO ──
    mp_result: dict = {}
    yolo_result: dict = {}

    m = _load_model()

    t_mp = threading.Thread(
        target=_run_mediapipe,
        args=(small_frame, mp_result),
    )
    t_yolo = threading.Thread(
        target=_run_yolo,
        args=(m, frame, yolo_result),
    )

    t_mp.start()
    t_yolo.start()

    t_mp.join()
    t_yolo.join()

    all_tips_small = mp_result.get("tips", [])
    hand_scale = mp_result.get("scale", 1.0)
    yolo_results = yolo_result.get("results", [])

    if not all_tips_small:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    all_fingertips = []
    for tips_small in all_tips_small:
        tips = tips_small.copy()
        tips[:, 0] = tips[:, 0] / scale
        tips[:, 1] = tips[:, 1] / scale
        all_fingertips.append(tips)

    roi_mask = _build_fingertip_roi_mask(all_fingertips, (h, w), hand_scale)

    combined_mask = _build_combined_mask(yolo_results, roi_mask, (h, w))

    if np.count_nonzero(combined_mask) == 0:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    processed_mask = _postprocess_mask(combined_mask)
    stabilized_mask, mean_error, flow_confident = _temporal_stabilize(processed_mask, small_gray, (h, w))

    # If optical flow confidence is low (too much motion or high error), rerun YOLO for a fresh mask
    if not flow_confident or mean_error > FLOW_ERROR_THRESHOLD:
        try:
            fresh_results = m.predict(frame, conf=YOLO_CONF, verbose=False)
        except Exception:
            fresh_results = []

        if len(fresh_results) > 0 and hasattr(fresh_results[0], "masks") and fresh_results[0].masks is not None:
            combined_mask = _build_combined_mask(fresh_results, roi_mask, (h, w))
            if np.count_nonzero(combined_mask) > 0:
                stabilized_mask = _postprocess_mask(combined_mask)

    if np.count_nonzero(stabilized_mask) == 0:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    result_bgr = _apply_color_transfer(frame, stabilized_mask)

    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ─────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────
@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/health")
def health() -> Any:
    with _model_lock:
        ready = _model_ready
    return jsonify({"status": "ok", "model_loaded": ready})


@app.route("/load_model", methods=["POST"])
def load_model() -> Any:
    try:
        _warmup_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/process_frame", methods=["POST"])
def process_frame() -> Any:
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400

    jpeg_bytes = file.read()
    if not jpeg_bytes:
        return jsonify({"error": "empty frame"}), 400

    try:
        output_bytes = _process_frame(jpeg_bytes)
    except Exception as exc:
        return jsonify({"error": f"processing failed: {str(exc)}"}), 500

    img_str = base64.b64encode(output_bytes).decode("utf-8")
    return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    _warmup_model()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)