"""
Safety analysis engine for SAFELABS.
Provides spatial IoU calculation, class detection rules, and event logging.
"""
import os
import time
import uuid
import random
import logging
import threading
from datetime import datetime
from typing import Any, List, Optional
from time_utils import get_myt_now
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Patch Ultralytics tracker callback to safely support alternating between .track() and .predict()
try:
    import ultralytics.trackers.track as _ultralytics_track

    _orig_postprocess_end = _ultralytics_track.on_predict_postprocess_end

    def _safe_on_predict_postprocess_end(predictor: Any, persist: bool = False) -> None:
        if not hasattr(predictor, "trackers") or not getattr(predictor, "trackers", None):
            return
        return _orig_postprocess_end(predictor, persist=persist)

    _ultralytics_track.on_predict_postprocess_end = _safe_on_predict_postprocess_end
except Exception:
    pass

from crypto_utils import encrypt_evidence, compute_evidence_hash
from hazard_engine import process_hazards, get_hazard_state

logger = logging.getLogger("safelabs.safety_engine")

# Global feature toggle: remove goggles from detection for now
ENABLE_GOGGLES_DETECTION = False


def _extract_boxes(yolo_output: Any) -> list:
    """Safely extract the box objects from an Ultralytics YOLO predict/track result."""
    if yolo_output is None:
        return []
    try:
        results_list = list(yolo_output)
        if not results_list:
            return []
        first_res = results_list[0]
        boxes = getattr(first_res, "boxes", None)
        return list(boxes) if boxes is not None else []
    except Exception:
        return []


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1: list/tuple [x1, y1, x2, y2]
        box2: list/tuple [x1, y1, x2, y2]

    Returns:
        float: IoU value in range [0.0, 1.0]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = float(box1_area + box2_area - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_current_ongoing_class(db_session, ClassSchedule):
    """
    Retrieve class schedule currently active based on current day, time, and semester dates.
    """
    now = get_myt_now()
    current_day = now.strftime('%A')
    current_time = now.time()
    current_date = now.date()

    candidates = db_session.query(ClassSchedule).filter_by(
        is_active=True
    ).filter(
        ClassSchedule.day_of_week == current_day,
        ClassSchedule.start_time <= current_time,
        ClassSchedule.end_time >= current_time,
    ).all()

    for c in candidates:
        if c.start_date and current_date < c.start_date:
            continue
        if c.end_date and current_date > c.end_date:
            continue
        return c

    return None


def is_monitoring_enabled_for_hazard(schedule, hazard_type):
    """
    Check if a hazard type is monitored based on class MonitoringConfig.
    If no class is ongoing, defaults to True (general laboratory monitoring).
    Combines face masks, lab coats, and gloves under Full PPE; goggles removed;
    monitors laptops and mobile phones.
    """
    if not schedule or not schedule.monitoring:
        return True

    cfg = schedule.monitoring
    h = hazard_type.lower()
    if 'phone' in h:
        return bool(getattr(cfg, 'monitor_phones', True))
    if 'laptop' in h:
        return bool(getattr(cfg, 'monitor_laptops', True))
    if 'mask' in h or 'coat' in h or 'glove' in h or 'ppe' in h:
        # Full PPE trigger
        return bool(getattr(cfg, 'monitor_full_ppe', True)) and bool(getattr(cfg, 'monitor_face_mask', True) or getattr(cfg, 'monitor_lab_coat', True))
    if 'goggle' in h or 'eye' in h:
        # Goggles detection removed from monitoring rules
        return False
    return True


def generate_confidence_score():
    """
    Generate a simulated AI confidence score between 0.72 and 0.98.
    In production, this will be replaced by the actual YOLOv8 model score.
    """
    return round(random.uniform(0.72, 0.98), 3)


def _stamp_evidence_frame(frame, hazard_type, severity, confidence_score):
    """
    Overlay metadata onto an evidence frame: timestamp, hazard type,
    severity badge, and confidence level.

    Returns a new frame (does not mutate the original).
    """
    stamped = frame.copy()
    h, w = stamped.shape[:2]
    now = get_myt_now()

    # Semi-transparent dark overlay bar at the bottom
    bar_height = 64
    overlay = stamped.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, stamped, 0.3, 0, stamped)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.42
    font_medium = 0.5
    white = (255, 255, 255)
    thickness = 1

    # Severity colour mapping
    severity_colors = {
        'critical': (60, 76, 231),    # Red (BGR)
        'high': (42, 127, 255),       # Orange
        'medium': (53, 183, 245),     # Amber
        'low': (219, 152, 56),        # Sky blue
    }
    sev_color = severity_colors.get(severity, white)

    # Row 1: Timestamp (left) + SAFELABS badge (right)
    ts_text = now.strftime('%d/%m/%y  %H:%M:%S')
    y_row1 = h - bar_height + 22
    cv2.putText(stamped, ts_text, (14, y_row1), font, font_small,
                white, thickness, cv2.LINE_AA)

    badge_text = "SAFELABS EVIDENCE"
    badge_size = cv2.getTextSize(badge_text, font, font_small, thickness)[0]
    cv2.putText(stamped, badge_text, (w - badge_size[0] - 14, y_row1),
                font, font_small, (180, 180, 180), thickness, cv2.LINE_AA)

    # Row 2: Hazard type + Severity + Confidence
    y_row2 = h - bar_height + 48
    hazard_label = hazard_type.upper()
    cv2.putText(stamped, hazard_label, (14, y_row2), font, font_medium,
                sev_color, thickness + 1, cv2.LINE_AA)

    hazard_size = cv2.getTextSize(hazard_label, font, font_medium,
                                  thickness + 1)[0]
    sev_x = 14 + hazard_size[0] + 16

    sev_text = f"[{severity.upper()}]"
    cv2.putText(stamped, sev_text, (sev_x, y_row2), font, font_small,
                sev_color, thickness, cv2.LINE_AA)

    if confidence_score is not None:
        conf_pct = f"{confidence_score * 100:.1f}%"
        sev_size = cv2.getTextSize(sev_text, font, font_small, thickness)[0]
        conf_x = sev_x + sev_size[0] + 16
        conf_label = f"CONF: {conf_pct}"
        cv2.putText(stamped, conf_label, (conf_x, y_row2), font,
                    font_small, (100, 255, 100), thickness, cv2.LINE_AA)

    # Top-left red recording dot
    cv2.circle(stamped, (20, 20), 6, (0, 0, 220), -1)
    cv2.putText(stamped, "REC", (32, 26), font, font_small,
                (0, 0, 220), thickness, cv2.LINE_AA)

    return stamped


def save_evidence_frame(frame, static_folder, prefix='violation',
                        hazard_type=None, severity=None,
                        confidence_score=None):
    """
    Save OpenCV frame as AES-256-GCM encrypted evidence in static/evidence/.
    Returns a tuple of (relative_path, sha256_hash).

    The file is saved with a .jpg.enc extension. The SHA-256 hash is
    computed on the plaintext JPEG data before encryption for
    chain-of-custody integrity verification.
    """
    evidence_dir = os.path.join(static_folder, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)

    timestamp_str = get_myt_now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{timestamp_str}_{uuid.uuid4().hex[:6]}.jpg.enc"
    abs_path = os.path.join(evidence_dir, filename)

    # Apply metadata stamp if detection context is provided
    if hazard_type and severity:
        frame = _stamp_evidence_frame(
            frame, hazard_type, severity, confidence_score
        )

    # Encode frame to JPEG bytes
    success, jpeg_buffer = cv2.imencode(
        '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not success:
        raise RuntimeError('Failed to encode evidence frame to JPEG')

    plaintext_bytes = jpeg_buffer.tobytes()

    # Compute integrity hash on plaintext before encryption
    evidence_hash = compute_evidence_hash(plaintext_bytes)

    # Encrypt and write to disk
    encrypted_bytes = encrypt_evidence(plaintext_bytes)
    with open(abs_path, 'wb') as f:
        f.write(encrypted_bytes)

    return f"evidence/{filename}", evidence_hash


def log_violation_event(
    db_session,
    ViolationLog,
    hazard_type,
    description=None,
    severity='medium',
    confidence_score=None,
    image_rel_path=None,
    evidence_hash=None,
    video_clip_path=None,
    class_schedule_id=None,
    camera_id=None,
):
    """
    Record a new violation in the database.
    """
    v = ViolationLog(
        hazard_type=hazard_type,
        description=description,
        severity=severity,
        confidence_score=confidence_score,
        image_path=image_rel_path,
        evidence_hash=evidence_hash,
        video_clip_path=video_clip_path,
        class_schedule_id=class_schedule_id,
        camera_id=camera_id,
    )
    db_session.add(v)
    db_session.commit()
    return v


def log_emergency_event(
    db_session,
    EmergencyAlert,
    alert_type,
    message,
    image_rel_path=None,
    evidence_hash=None,
    camera_id=None,
):
    """
    Record a critical emergency alert (fire, smoke, gas, spill).
    """
    alert = EmergencyAlert(
        alert_type=alert_type,
        message=message,
        image_path=image_rel_path,
        evidence_hash=evidence_hash,
        camera_id=camera_id,
    )
    db_session.add(alert)
    db_session.commit()
    return alert


# ===================================================================
#  ROBOFLOW PPE SAFETY MODEL INTEGRATION (Roboflow API + Cloud API)
# ===================================================================
AVAILABLE_MODELS = {
    "roboflow": {
        "id": "roboflow",
        "name": "Roboflow API",
        "full_name": "Roboflow API (general-segmentation-api-4)",
        "file": None,
        "path": None,
        "type": "api",
        "description": "Roboflow Serverless Segmentation Workflow",
        "badge": "Roboflow API",
    },
    "cloud_api": {
        "id": "cloud_api",
        "name": "Cloud API",
        "full_name": "Roboflow Cloud Logic (ppe-chemistry-lab-vppe-chemistry-lab-roc3r-1-yolo26n-t1-logic)",
        "file": None,
        "path": None,
        "type": "cloud_api",
        "description": "Roboflow Serverless Workflow (YOLO26n Logic)",
        "badge": "Cloud API",
    },
    "hybrid": {
        "id": "hybrid",
        "name": "Hybrid",
        "full_name": "Hybrid (Local YOLO Tracking + Roboflow API Detection)",
        "file": None,
        "path": None,
        "type": "hybrid",
        "description": "Fast local person tracking (every frame) fused with high-quality Roboflow PPE detection (periodic async API calls)",
        "badge": "HYBRID",
    },
}


def normalize_backend_id(val: str) -> str:
    """Normalize model identifiers to canonical model keys ('hybrid', 'roboflow', 'cloud_api')."""
    if not val:
        return "hybrid"
    c = str(val).strip().lower()
    mapping = {
        # Hybrid: fast local tracking + periodic API detection
        "hybrid": "hybrid",
        "merged": "hybrid",
        "fusion": "hybrid",
        "dual": "hybrid",
        "roboflow": "roboflow",
        "roboflow_api": "roboflow",
        "api": "roboflow",
        "general-segmentation-api-4": "roboflow",
        "cloud_api": "cloud_api",
        "cloud": "cloud_api",
        "roboflow_cloud": "cloud_api",
        "serverless": "cloud_api",
        "yolo26n": "cloud_api",
        "logic": "cloud_api",
        "webrtc": "cloud_api",
        # Legacy fallbacks:
        "best_1": "roboflow",
        "best(1)": "roboflow",
        "best_2": "roboflow",
        "best(2)": "roboflow",
        "best_3": "roboflow",
        "best(3)": "roboflow",
        "best_4": "roboflow",
        "best(4)": "roboflow",
        "local_yolo": "roboflow",
        "local": "roboflow",
    }
    return mapping.get(c, "hybrid")


_yolo_models = {}
_yolo_model = None  # Backward compatibility reference
_model_lock = threading.Lock()

# Active backend state: default to 'hybrid'
_active_backend = normalize_backend_id(os.environ.get("SAFELABS_DETECTION_BACKEND", "hybrid"))
if _active_backend not in AVAILABLE_MODELS:
    _active_backend = "hybrid"


def get_active_backend() -> str:
    """Return the currently active detection backend identifier ('roboflow', 'cloud_api')."""
    global _active_backend
    return _active_backend


def set_active_backend(backend: str) -> str:
    """Switch active detection backend with zero downtime."""
    global _active_backend
    target = normalize_backend_id(backend)
    if target in AVAILABLE_MODELS:
        _active_backend = target
        logger.info(f"SAFELABS detection backend switched to: {_active_backend} ({AVAILABLE_MODELS[target]['name']})")
        return _active_backend
    raise ValueError(f"Invalid backend '{backend}'. Supported: {list(AVAILABLE_MODELS.keys())}")


def get_available_models():
    """Return list of all configured model backends with status."""
    active = get_active_backend()
    result = []
    for k, v in AVAILABLE_MODELS.items():
        item = dict(v)
        item["is_active"] = (k == active)
        if item["type"] == "local_yolo":
            item["exists"] = os.path.exists(item["path"]) if item.get("path") else False
        else:
            item["exists"] = True
        result.append(item)
    return result


def get_yolo_model(model_identifier=None):
    """
    Lazy load and cache YOLOv8 models.
    Supports model identifier ('best_1', 'best_2', 'best_3', 'best_4'), file path,
    or defaults to the currently active model.
    """
    global _yolo_models, _yolo_model
    with _model_lock:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics library is not installed in the environment.")
            return None

        target_id = normalize_backend_id(model_identifier) if model_identifier else normalize_backend_id(_active_backend)
        if target_id in ("roboflow", "cloud_api"):
            target_id = "best_4"

        if target_id in _yolo_models and _yolo_models[target_id] is not None:
            _yolo_model = _yolo_models[target_id]
            return _yolo_models[target_id]

        model_path = None
        if target_id in AVAILABLE_MODELS and AVAILABLE_MODELS[target_id].get("path"):
            model_path = AVAILABLE_MODELS[target_id]["path"]
        elif model_identifier and os.path.exists(model_identifier):
            model_path = model_identifier
        else:
            candidates = [
                os.path.join(os.path.dirname(__file__), "models", "best(4).pt"),
                os.path.join(os.path.dirname(__file__), "models", "best(3).pt"),
                os.path.join(os.path.dirname(__file__), "models", "best(2).pt"),
                os.path.join(os.path.dirname(__file__), "models", "best(1).pt"),
                os.path.join(os.path.dirname(__file__), "models", "best.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"No YOLO model file found for target '{target_id}' at {model_path}.")
            return None

        try:
            resolved = os.path.abspath(model_path)
            logger.info(f"Loading YOLO PPE model weights [{target_id}] from: {resolved}")
            loaded_model = YOLO(resolved)
            _yolo_models[target_id] = loaded_model
            _yolo_model = loaded_model
            logger.info(f"YOLO model [{target_id}] loaded successfully. Classes: {loaded_model.names}")
            return loaded_model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            return None


_roboflow_client = None
_roboflow_lock = threading.Lock()
_roboflow_init_attempted = False


def get_roboflow_client():
    """
    Lazy load and cache Roboflow Inference HTTP client with header authentication.
    """
    global _roboflow_client, _roboflow_init_attempted
    with _roboflow_lock:
        if _roboflow_client is not None:
            return _roboflow_client
        if _roboflow_init_attempted:
            return None
        _roboflow_init_attempted = True
        try:
            import importlib
            inf_sdk = importlib.import_module("inference_sdk")
            InferenceHTTPClient = inf_sdk.InferenceHTTPClient
            InferenceConfiguration = inf_sdk.InferenceConfiguration
            api_url = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
            api_key = os.environ.get("ROBOFLOW_API_KEY", "YCqOfbCT5u8IfxK6smNX")
            _roboflow_client = InferenceHTTPClient(
                api_url=api_url,
                api_key=api_key,
            ).configure(InferenceConfiguration(api_key_transport="header"))
            logger.info("Roboflow Serverless Inference Client initialized successfully.")
            return _roboflow_client
        except Exception as e:
            logger.warning(f"Roboflow inference client unavailable (falling back to local models): {e}")
            return None


_person_model = None
_person_lock = threading.Lock()


def get_person_model():
    """
    Lazy load and cache standard YOLOv8n person detector model.
    """
    global _person_model
    with _person_lock:
        if _person_model is not None:
            return _person_model

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics library is not installed in the environment.")
            return None

        candidates = [
            os.environ.get("SAFELABS_PERSON_MODEL_PATH"),
            os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt"),
            "yolov8n.pt",
        ]
        resolved = None
        for c in candidates:
            if c and os.path.exists(c):
                resolved = os.path.abspath(c)
                break
        if not resolved:
            resolved = "yolov8n.pt"

        try:
            logger.info(f"Loading YOLO person model from: {resolved}")
            _person_model = YOLO(resolved)
            return _person_model
        except Exception as e:
            logger.error(f"Failed to load person model from {resolved}: {e}")
            return None


# ===================================================================
#  CONTRABAND DETECTION ENGINE (bags, drinks, food, electronics on workbench)
# ===================================================================
# coco class ids for lab contraband
CONTRABAND_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    45: "bowl",
    47: "apple",
    48: "sandwich",
    53: "pizza",
    63: "laptop",
    67: "cell phone",
}

# safety_engine.py - class-specific confidence gates
CLASS_CONFIDENCE_THRESHOLDS = {
    63: 0.35,  # Laptop: clear footprint, reliable at 0.35
    67: 0.28,  # Cell Phone: lowered to 0.28 to catch dark phones on black surfaces
    "default": 0.32,
}

# bench table boundary (default polygon for lab workbench surface)
# configurable at runtime via set_bench_roi() or env SAFELABS_BENCH_ROI
DEFAULT_BENCH_ROI = np.array([
    [150, 420],
    [1180, 420],
    [1260, 720],
    [60, 720],
], dtype=np.int32)


def _init_bench_roi():
    # allow env override: "x1,y1,x2,y2..." or json "[[x1,y1],...]"
    raw = os.environ.get("SAFELABS_BENCH_ROI", "").strip()
    if not raw:
        return DEFAULT_BENCH_ROI
    try:
        import json
        if raw.startswith("["):
            return np.array(json.loads(raw), dtype=np.int32)
        pts = [int(v.strip()) for v in raw.replace(";", ",").split(",") if v.strip()]
        if len(pts) >= 6 and len(pts) % 2 == 0:
            return np.array([[pts[i], pts[i + 1]] for i in range(0, len(pts), 2)], dtype=np.int32)
    except Exception:
        pass
    return DEFAULT_BENCH_ROI


BENCH_ROI = _init_bench_roi()


def set_bench_roi(points):
    # runtime update for workbench zone
    global BENCH_ROI
    BENCH_ROI = np.array(points, dtype=np.int32)


_contraband_lock = threading.Lock()
_contraband_frame_counter = 0
_contraband_cached_items = None
CONTRABAND_INTERVAL = 5  # run heavy coco check every 5th frame


def detect_contraband(frame, conf_threshold=None, imgsz=640, roi=None):
    # run yolov8n on contraband classes only
    model = get_person_model()
    if model is None or frame is None:
        return []

    bench_polygon = roi if roi is not None else BENCH_ROI
    items = []

    # Lowest gate among configured classes to avoid pre-filtering valid items in YOLO pass
    min_gate = min(v for v in CLASS_CONFIDENCE_THRESHOLDS.values() if isinstance(v, (int, float)))
    predict_conf = min(min_gate, conf_threshold) if conf_threshold is not None else min_gate

    try:
        results = model.predict(
            frame,
            classes=list(CONTRABAND_CLASSES.keys()),
            conf=predict_conf,
            imgsz=imgsz,
            verbose=False,
        )
        boxes = _extract_boxes(results)
        if not boxes:
            return []

        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])

            # Enforce class-specific confidence gate
            required_conf = CLASS_CONFIDENCE_THRESHOLDS.get(
                cls_id, CLASS_CONFIDENCE_THRESHOLDS.get("default", 0.32)
            )
            if conf < required_conf:
                continue

            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]

            # bottom-center foot of box
            x_center = (x1 + x2) / 2.0
            y_max = float(y2)

            # anchor inside bench zone check
            if cv2.pointPolygonTest(bench_polygon, (x_center, y_max), False) >= 0:
                items.append({
                    "class_id": cls_id,
                    "class_name": CONTRABAND_CLASSES.get(cls_id, f"item_{cls_id}"),
                    "confidence": round(conf, 3),
                    "box": [x1, y1, x2, y2],
                    "anchor": [round(x_center, 1), round(y_max, 1)],
                    "on_bench": True,
                })
    except Exception as e:
        logger.error(f"contraband detection error: {e}")

    return items


def get_cached_contraband(frame, conf_threshold=None, imgsz=640, force_run=False):
    # skip heavy coco run on filler frames, reuse cached boxes to eliminate hud flicker
    global _contraband_frame_counter, _contraband_cached_items
    with _contraband_lock:
        _contraband_frame_counter += 1
        should_run = force_run or (_contraband_frame_counter % CONTRABAND_INTERVAL == 0) or (_contraband_cached_items is None)

    if should_run:
        fresh = detect_contraband(frame, conf_threshold=conf_threshold, imgsz=imgsz)
        with _contraband_lock:
            _contraband_cached_items = fresh
            return list(_contraband_cached_items)
    else:
        with _contraband_lock:
            return list(_contraband_cached_items or [])


# ===================================================================
#  HYBRID BACKEND: Async API Cache + Person Tracking Fusion
# ===================================================================
# Global cache for the hybrid backend's periodic Roboflow API results.
# The API is called asynchronously every N frames to avoid blocking the
# real-time person tracking loop. Between API calls, the most recent
# PPE detections are carried forward and spatially fused with fresh
# person bounding boxes from the fast local YOLO model.

_hybrid_cache_lock = threading.Lock()
_hybrid_cached_ppe = []            # Most recent API PPE detections
_hybrid_cache_timestamp = 0.0      # When the cache was last updated
_hybrid_api_in_flight = False      # Prevent overlapping API calls
_hybrid_frame_counter = 0          # Frame counter for API call cadence
_hybrid_api_latency_ms = 0.0       # Last API round-trip time
_hybrid_active_persons = {}        # track_id -> {'box': box, 'conf': conf, 'lost_frames': int} (anti-flicker smoothing)

# Tuning: how often to call the API (every N analyze_frame_ppe calls in hybrid mode)
HYBRID_API_INTERVAL = int(os.environ.get("SAFELABS_HYBRID_API_INTERVAL", "8"))
# Maximum age (seconds) before cached API results are considered stale
HYBRID_CACHE_TTL = float(os.environ.get("SAFELABS_HYBRID_CACHE_TTL", "3.0"))


def _hybrid_api_worker(frame, ref_persons=None):
    """
    Background thread worker: calls Roboflow API on a frame and updates
    the global PPE detection cache with person-relative body anchors.
    Non-blocking to the main inference loop.
    """
    global _hybrid_cached_ppe, _hybrid_cache_timestamp, _hybrid_api_in_flight, _hybrid_api_latency_ms

    try:
        api_start = time.time()
        result = run_roboflow_ppe_inference(frame)
        api_elapsed = (time.time() - api_start) * 1000

        all_dets = result.get("detections", [])
        # Filter to PPE items only (persons come from local YOLO)
        allowed_ppe = ("gloves", "lab_coat", "mask") if not ENABLE_GOGGLES_DETECTION else ("goggles", "gloves", "lab_coat", "mask")
        ppe_only = [
            d for d in all_dets
            if d["class_name"] in allowed_ppe
        ]

        # Calculate relative spatial offsets against reference persons present when frame was captured
        ref_list = ref_persons or []
        for d in ppe_only:
            ibox = d["box"]
            cx = (ibox[0] + ibox[2]) / 2.0
            cy = (ibox[1] + ibox[3]) / 2.0
            owner_tid = None
            rel_box = None

            for p in ref_list:
                px1, py1, px2, py2 = p["box"]
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)

                # Check if item falls inside person's body or proximity bounds
                if (px1 - 0.25 * pw <= cx <= px2 + 0.25 * pw) and (py1 - 0.15 * ph <= cy <= py2 + 0.15 * ph):
                    owner_tid = p.get("track_id")
                    rel_box = [
                        (ibox[0] - px1) / pw,
                        (ibox[1] - py1) / ph,
                        (ibox[2] - px1) / pw,
                        (ibox[3] - py1) / ph,
                    ]
                    break

            d["owner_track_id"] = owner_tid
            d["rel_box"] = rel_box

        with _hybrid_cache_lock:
            _hybrid_cached_ppe = ppe_only
            _hybrid_cache_timestamp = time.time()
            _hybrid_api_latency_ms = round(api_elapsed, 1)

        logger.info(
            f"[HYBRID] API refresh complete: {len(ppe_only)} PPE items anchored "
            f"in {api_elapsed:.0f}ms (total API detections: {len(all_dets)})"
        )
    except Exception as e:
        logger.error(f"[HYBRID] Async API worker error: {e}")
    finally:
        with _hybrid_cache_lock:
            _hybrid_api_in_flight = False


def _hybrid_fire_api(frame, ref_persons=None):
    """
    Launch a background thread to call the Roboflow API if one isn't
    already in flight. The frame and reference person boxes are copied
    to avoid race conditions.
    """
    global _hybrid_api_in_flight
    with _hybrid_cache_lock:
        if _hybrid_api_in_flight:
            return  # Already running, skip
        _hybrid_api_in_flight = True

    copied_persons = [dict(p) for p in (ref_persons or [])]
    thread = threading.Thread(
        target=_hybrid_api_worker,
        args=(frame.copy(), copied_persons),
        daemon=True,
        name="HybridAPIWorker",
    )
    thread.start()


def _hybrid_get_cached_ppe():
    """
    Return the cached PPE detections and their age. If the cache is older
    than HYBRID_CACHE_TTL, returns an empty list (stale data discarded).
    """
    with _hybrid_cache_lock:
        age = time.time() - _hybrid_cache_timestamp
        if age > HYBRID_CACHE_TTL and _hybrid_cache_timestamp > 0:
            # Cache expired but keep data if API is currently refreshing
            if _hybrid_api_in_flight:
                return list(_hybrid_cached_ppe), age, _hybrid_api_latency_ms
            return [], age, _hybrid_api_latency_ms
        return list(_hybrid_cached_ppe), age, _hybrid_api_latency_ms


def run_roboflow_ppe_inference(frame, classes=None):
    """
    Execute Roboflow Serverless Workflow PPE segmentation & detection on an OpenCV BGR frame.
    Returns:
        dict with:
            - 'detections': list of dicts (class_name, raw_class, confidence, box: [x1, y1, x2, y2])
            - 'annotated_base64': str or None
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {"detections": [], "annotated_base64": None}

    client = get_roboflow_client()
    if client is None:
        return {"detections": [], "annotated_base64": None}

    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "balaji-rao-kkesz")
    workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "general-segmentation-api-4")
    target_classes = classes or os.environ.get(
        "ROBOFLOW_CLASSES", "gloves, lab coat, Lab Goggles, PPE-CHEMISTRY-LAB, person, mask"
    )

    try:
        h, w = frame.shape[:2]
        # Optimize network upload speed by resizing (default 640px for faster transfer)
        max_upload_w = int(os.environ.get("ROBOFLOW_MAX_UPLOAD_WIDTH", "640"))
        if w > max_upload_w:
            scale = float(max_upload_w) / w
            upload_frame = cv2.resize(frame, (max_upload_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            scale_x = w / float(max_upload_w)
            scale_y = h / float(int(h * scale))
        else:
            upload_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        result = client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow_id,
            images={"image": upload_frame},
            parameters={"classes": target_classes},
            use_cache=True,
        )

        detections = []
        annotated_b64 = None

        if result and len(result) > 0:
            entry = result[0]
            annotated_b64 = entry.get("annotated_image")
            preds_block = entry.get("predictions", {})
            pred_list = preds_block.get("predictions", []) if isinstance(preds_block, dict) else []

            for p in pred_list:
                cx = float(p.get("x", 0)) * scale_x
                cy = float(p.get("y", 0)) * scale_y
                pw = float(p.get("width", 0)) * scale_x
                ph = float(p.get("height", 0)) * scale_y

                x1 = max(0, int(cx - pw / 2.0))
                y1 = max(0, int(cy - ph / 2.0))
                x2 = min(w, int(cx + pw / 2.0))
                y2 = min(h, int(cy + ph / 2.0))

                raw_class = str(p.get("class", "")).strip()
                cname = raw_class.lower()

                if "goggle" in cname:
                    if not ENABLE_GOGGLES_DETECTION:
                        continue
                    norm_class = "goggles"
                elif "glove" in cname:
                    norm_class = "gloves"
                elif "coat" in cname:
                    norm_class = "lab_coat"
                elif "mask" in cname:
                    norm_class = "mask"
                elif "person" in cname or "ppe-chemistry" in cname:
                    norm_class = "person"
                else:
                    norm_class = cname.replace(" ", "_")

                conf = float(p.get("confidence", 0.0))

                detections.append({
                    "class_name": norm_class,
                    "raw_class": raw_class,
                    "confidence": round(conf, 3),
                    "box": [x1, y1, x2, y2],
                    "rle_mask": p.get("rle_mask"),
                })

        return {"detections": detections, "annotated_base64": annotated_b64}

    except Exception as e:
        logger.error(f"Roboflow workflow execution error: {e}")
        return {"detections": [], "annotated_base64": None}


def run_cloud_api_ppe_inference(frame):
    """
    Execute Roboflow Cloud Workflow (YOLO26n Logic) on an OpenCV BGR frame.
    Workflow: ppe-chemistry-lab-vppe-chemistry-lab-roc3r-1-yolo26n-t1-logic
    Workspace: balaji-rao-kkesz
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {"detections": [], "annotated_base64": None}

    client = get_roboflow_client()
    if client is None:
        return {"detections": [], "annotated_base64": None}

    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "balaji-rao-kkesz")
    workflow_id = os.environ.get(
        "ROBOFLOW_CLOUD_WORKFLOW_ID",
        "ppe-chemistry-lab-vppe-chemistry-lab-roc3r-1-yolo26n-t1-logic"
    )

    try:
        h, w = frame.shape[:2]
        max_upload_w = int(os.environ.get("ROBOFLOW_MAX_UPLOAD_WIDTH", "640"))
        if w > max_upload_w:
            scale = float(max_upload_w) / w
            upload_frame = cv2.resize(frame, (max_upload_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            scale_x = w / float(max_upload_w)
            scale_y = h / float(int(h * scale))
        else:
            upload_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        result = client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow_id,
            images={"image": upload_frame},
            use_cache=True,
        )

        detections = []
        annotated_b64 = None

        if result and len(result) > 0:
            entry = result[0]
            annotated_b64 = entry.get("annotated_image")
            preds_block = entry.get("predictions", {})
            pred_list = preds_block.get("predictions", []) if isinstance(preds_block, dict) else (preds_block if isinstance(preds_block, list) else [])

            for p in pred_list:
                cx = float(p.get("x", 0)) * scale_x
                cy = float(p.get("y", 0)) * scale_y
                pw = float(p.get("width", 0)) * scale_x
                ph = float(p.get("height", 0)) * scale_y

                x1 = max(0, int(cx - pw / 2.0))
                y1 = max(0, int(cy - ph / 2.0))
                x2 = min(w, int(cx + pw / 2.0))
                y2 = min(h, int(cy + ph / 2.0))

                raw_class = str(p.get("class", p.get("class_name", ""))).strip()
                cname = raw_class.lower()

                if "goggle" in cname:
                    if not ENABLE_GOGGLES_DETECTION:
                        continue
                    norm_class = "goggles"
                elif "glove" in cname:
                    norm_class = "gloves"
                elif "coat" in cname:
                    norm_class = "lab_coat"
                elif "mask" in cname:
                    norm_class = "mask"
                elif "person" in cname or "ppe-chemistry" in cname:
                    norm_class = "person"
                else:
                    norm_class = cname.replace(" ", "_")

                conf = float(p.get("confidence", 0.0))

                detections.append({
                    "class_name": norm_class,
                    "raw_class": raw_class,
                    "confidence": round(conf, 3),
                    "box": [x1, y1, x2, y2],
                    "rle_mask": p.get("rle_mask"),
                })

        return {"detections": detections, "annotated_base64": annotated_b64}

    except Exception as e:
        logger.error(f"Roboflow cloud workflow execution error: {e}")
        return {"detections": [], "annotated_base64": None}


def run_yolo_inference(frame, conf_threshold=0.25, model_identifier=None):
    """
    Primary PPE detection runner. Uses active backend or specified model ('best_1'..'best_4', 'roboflow', 'cloud_api').
    """
    backend = normalize_backend_id(model_identifier or get_active_backend())
    if backend in ("best_1", "best_2", "best_3", "best_4"):
        model = get_yolo_model(backend)
        if model is None:
            return []
        try:
            results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False)
            detections = []
            for b in _extract_boxes(results):
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = [round(float(v), 1) for v in b.xyxy[0]]
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "box": xyxy,
                })
            return detections
        except Exception as e:
            logger.error(f"Error during local YOLO inference ({backend}): {e}")
            return []
    elif backend == "cloud_api":
        res = run_cloud_api_ppe_inference(frame)
        dets = res.get("detections", [])
        if dets:
            return [d for d in dets if d["confidence"] >= conf_threshold]
        return []
    else:
        res = run_roboflow_ppe_inference(frame)
        dets = res.get("detections", [])
        if dets:
            return [d for d in dets if d["confidence"] >= conf_threshold]
        return []


def _box_in_zone(box, zone, min_overlap=0.08):
    """Check if center of box falls inside zone or has IoU overlap."""
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0

    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
        return True

    xA = max(bx1, zx1)
    yA = max(by1, zy1)
    xB = min(bx2, zx2)
    yB = min(by2, zy2)
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return (inter_area / box_area) >= min_overlap


_hazard_frame_counter = 0
_hazard_lock = threading.Lock()
_last_hazard_log_time = 0.0


def analyze_frame_ppe(frame, person_conf=0.25, ppe_conf=0.15, imgsz=640, backend_override=None, frame_idx=None):
    """
    Unified Person + PPE Detection & Compliance Engine.
    Supports dual backends:
      1. 'local_yolo': Local YOLOv8 PPE model (models/best.pt) + person detector (models/yolov8n.pt)
      2. 'roboflow': Roboflow Serverless Workflow (general-segmentation-api-4)
    Detects humans and matches PPE items (goggles, gloves, lab coat, mask)
    against spatial body regions (head, torso, hands).
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {
            "timestamp": time.time(),
            "persons_count": 0,
            "persons": [],
            "standalone_ppe": [],
            "contraband": [],
            "contraband_count": 0,
            "fire_active": False,
            "smoke_active": False,
            "hazards": [],
            "has_violations": False,
            "has_contraband": False,
            "has_emergency": False,
            "model_backend": get_active_backend(),
            "latency_ms": 0.0,
        }

    global _hazard_frame_counter, _last_hazard_log_time
    with _hazard_lock:
        _hazard_frame_counter += 1
        current_frame_idx = frame_idx if frame_idx is not None else _hazard_frame_counter

    start_time = time.time()
    backend = normalize_backend_id(backend_override or get_active_backend())
    if backend not in AVAILABLE_MODELS:
        backend = "best_4"

    h, w = frame.shape[:2]
    person_boxes = []
    ppe_items = []
    annotated_b64 = None

    if backend in ("best_1", "best_2", "best_3", "best_4"):
        # 1a. Detect persons using local YOLOv8n
        person_model = get_person_model()
        if person_model is not None:
            try:
                res_p = person_model.predict(
                    frame, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False
                )
                for b in _extract_boxes(res_p):
                    person_boxes.append({
                        "box": [int(v) for v in b.xyxy[0]],
                        "conf": round(float(b.conf[0]), 3),
                        "source": "local_yolov8n",
                    })
            except Exception as e:
                logger.error(f"Local person detector error: {e}")

        # 1b. Detect PPE items using requested local model
        ppe_model = get_yolo_model(backend)
        if ppe_model is not None:
            try:
                res_ppe = ppe_model.predict(
                    frame, conf=ppe_conf, imgsz=imgsz, verbose=False
                )
                for b in _extract_boxes(res_ppe):
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    xyxy = [int(v) for v in b.xyxy[0]]
                    raw_name = ppe_model.names.get(cls_id, f"class_{cls_id}")
                    norm_name = raw_name.lower().replace(" ", "_")
                    if "goggle" in norm_name:
                        if not ENABLE_GOGGLES_DETECTION:
                            continue
                        cname = "goggles"
                    elif "glove" in norm_name:
                        cname = "gloves"
                    elif "coat" in norm_name or "lab_coat" in norm_name:
                        cname = "lab_coat"
                    elif "mask" in norm_name:
                        cname = "mask"
                    else:
                        cname = norm_name

                    ppe_items.append({
                        "class_id": cls_id,
                        "class_name": cname,
                        "raw_class": raw_name,
                        "confidence": round(conf, 3),
                        "box": xyxy,
                    })
            except Exception as e:
                logger.error(f"Local YOLO PPE inference error ({backend}): {e}")

        backend_label = AVAILABLE_MODELS[backend]["full_name"]

    elif backend == "cloud_api":
        # Roboflow Cloud Workflow (ppe-chemistry-lab-vppe-chemistry-lab-roc3r-1-yolo26n-t1-logic)
        cloud_res = run_cloud_api_ppe_inference(frame)
        all_detections = cloud_res.get("detections", [])
        annotated_b64 = cloud_res.get("annotated_base64")

        # Extract persons from Cloud API if present
        for d in all_detections:
            if d["class_name"] == "person" and d["confidence"] >= person_conf:
                person_boxes.append({
                    "box": d["box"],
                    "conf": d["confidence"],
                    "source": "cloud_api",
                })

        # Fallback to local person detector if Cloud API found no persons
        if not person_boxes:
            person_model = get_person_model()
            if person_model is not None:
                try:
                    res_p = person_model.predict(
                        frame, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False
                    )
                    for b in _extract_boxes(res_p):
                        person_boxes.append({
                            "box": [int(v) for v in b.xyxy[0]],
                            "conf": round(float(b.conf[0]), 3),
                            "source": "local_person_fallback",
                        })
                except Exception as e:
                    logger.error(f"Local person fallback error: {e}")

        allowed_ppe = ("gloves", "lab_coat", "mask") if not ENABLE_GOGGLES_DETECTION else ("goggles", "gloves", "lab_coat", "mask")
        ppe_items = [
            d for d in all_detections
            if d["class_name"] in allowed_ppe and d["confidence"] >= ppe_conf
        ]
        backend_label = AVAILABLE_MODELS["cloud_api"]["full_name"]

    elif backend == "hybrid":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  HYBRID MODE: Ultra-fast ByteTrack tracking + async API PPE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 1: ALWAYS run fast ByteTrack person tracking (every frame)
        global _hybrid_frame_counter
        _hybrid_frame_counter += 1

        person_model = get_person_model()
        if person_model is not None:
            # Resolve custom ByteTrack config for stable persistent IDs across turns and occlusions
            custom_tracker = os.path.join(os.path.dirname(__file__), "safelabs_tracker.yaml")
            tracker_cfg = custom_tracker if os.path.exists(custom_tracker) else "bytetrack.yaml"

            try:
                # Use ByteTrack for temporal coherence, Kalman-filter smoothing, and persistent IDs
                res_p = person_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=person_conf,
                    imgsz=imgsz,
                    verbose=False,
                    tracker=tracker_cfg,
                )
                for b in _extract_boxes(res_p):
                    tid = int(b.id[0]) if getattr(b, "id", None) is not None and b.id is not None else None
                    person_boxes.append({
                        "box": [int(v) for v in b.xyxy[0]],
                        "conf": round(float(b.conf[0]), 3),
                        "track_id": tid,
                        "source": "hybrid_bytetrack",
                    })
            except Exception as e:
                # Resilient fallback to predict if tracker encounters a transient state
                try:
                    res_p = person_model.predict(
                        frame, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False
                    )
                    for b in _extract_boxes(res_p):
                        person_boxes.append({
                            "box": [int(v) for v in b.xyxy[0]],
                            "conf": round(float(b.conf[0]), 3),
                            "track_id": None,
                            "source": "hybrid_local_fallback",
                        })
                except Exception as ex:
                    logger.error(f"[HYBRID] Local person tracking error: {ex}")

        # Kalman anti-flicker smoothing: bridge micro-drops (<=2 frames) to eliminate visual bounding box flicker
        current_seen_tids = set()
        for p in person_boxes:
            tid = p.get("track_id")
            if tid is not None:
                current_seen_tids.add(tid)
                _hybrid_active_persons[tid] = {
                    "box": p["box"],
                    "conf": p["conf"],
                    "track_id": tid,
                    "lost_frames": 0,
                    "source": p.get("source", "hybrid_bytetrack"),
                }

        stale_tids = []
        for tid, pdata in _hybrid_active_persons.items():
            if tid not in current_seen_tids:
                pdata["lost_frames"] += 1
                if pdata["lost_frames"] <= 2:
                    person_boxes.append({
                        "box": pdata["box"],
                        "conf": round(pdata["conf"] * 0.95, 3),
                        "track_id": tid,
                        "source": "grace_smoothing",
                    })
                else:
                    stale_tids.append(tid)

        for tid in stale_tids:
            _hybrid_active_persons.pop(tid, None)

        # Step 2: PERIODICALLY fire async API call (every Nth frame, non-blocking)
        if _hybrid_frame_counter % HYBRID_API_INTERVAL == 0:
            _hybrid_fire_api(frame, ref_persons=person_boxes)

        # Step 3: Retrieve cached PPE detections and project them dynamically with tracked persons
        cached_ppe, cache_age, api_lat = _hybrid_get_cached_ppe()

        # Build lookup for currently active persons
        person_by_track = {p["track_id"]: p for p in person_boxes if p.get("track_id") is not None}
        projected_ppe = []

        for item in cached_ppe:
            item_copy = dict(item)
            owner_tid = item.get("owner_track_id")
            rel_box = item.get("rel_box")

            if owner_tid is not None and owner_tid in person_by_track and rel_box:
                # Dynamic Body Projection: Person has moved, update PPE box to their new live coordinates!
                curr_p = person_by_track[owner_tid]
                c_px1, c_py1, c_px2, c_py2 = curr_p["box"]
                c_pw = max(1, c_px2 - c_px1)
                c_ph = max(1, c_py2 - c_py1)

                rx1, ry1, rx2, ry2 = rel_box
                p_ix1 = max(0, min(w, int(c_px1 + rx1 * c_pw)))
                p_iy1 = max(0, min(h, int(c_py1 + ry1 * c_ph)))
                p_ix2 = max(0, min(w, int(c_px1 + rx2 * c_pw)))
                p_iy2 = max(0, min(h, int(c_py1 + ry2 * c_ph)))

                item_copy["box"] = [p_ix1, p_iy1, p_ix2, p_iy2]
                item_copy["is_anchored"] = True
            elif not item.get("owner_track_id"):
                # Tabletop / unanchored item: stays in place on bench
                item_copy["is_anchored"] = False

            projected_ppe.append(item_copy)

        ppe_items = [
            d for d in projected_ppe
            if d.get("confidence", 0) >= ppe_conf
        ]

        # Build the backend label with cache status info
        cache_status = "LIVE" if cache_age < 1.0 else f"{cache_age:.1f}s ago"
        with _hybrid_cache_lock:
            in_flight = _hybrid_api_in_flight
        flight_tag = " | API↻" if in_flight else ""
        anchored_cnt = sum(1 for d in ppe_items if d.get("is_anchored"))
        backend_label = (
            f"HYBRID (Local YOLO + Roboflow API | "
            f"PPE: {len(ppe_items)} ({anchored_cnt} body-locked) | {cache_status}{flight_tag})"
        )

    else:
        # Roboflow Serverless Workflow (general-segmentation-api-4)
        roboflow_res = run_roboflow_ppe_inference(frame)
        all_detections = roboflow_res.get("detections", [])
        annotated_b64 = roboflow_res.get("annotated_base64")

        # Extract persons from Roboflow
        for d in all_detections:
            if d["class_name"] == "person" and d["confidence"] >= person_conf:
                person_boxes.append({
                    "box": d["box"],
                    "conf": d["confidence"],
                    "source": "roboflow",
                })

        # Fallback to local person detector if Roboflow found no persons
        if not person_boxes:
            person_model = get_person_model()
            if person_model is not None:
                try:
                    res_p = person_model.predict(
                        frame, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False
                    )
                    for b in _extract_boxes(res_p):
                        person_boxes.append({
                            "box": [int(v) for v in b.xyxy[0]],
                            "conf": round(float(b.conf[0]), 3),
                            "source": "local_person_fallback",
                        })
                except Exception as e:
                    logger.error(f"Local person fallback error: {e}")

        allowed_ppe = ("gloves", "lab_coat", "mask") if not ENABLE_GOGGLES_DETECTION else ("goggles", "gloves", "lab_coat", "mask")
        ppe_items = [
            d for d in all_detections
            if d["class_name"] in allowed_ppe and d["confidence"] >= ppe_conf
        ]
        backend_label = AVAILABLE_MODELS["roboflow"]["full_name"]

    # 2. Spatial Association: Associate PPE items with detected persons
    analyzed_persons = []
    claimed_ppe = set()

    for idx, p in enumerate(person_boxes):
        px1, py1, px2, py2 = p["box"]
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        head_zone = [px1, py1, px2, int(py1 + 0.35 * ph)]
        torso_zone = [px1, int(py1 + 0.18 * ph), px2, int(py1 + 0.85 * ph)]
        hands_zone = [max(0, px1 - int(0.25 * pw)), int(py1 + 0.35 * ph), min(w, px2 + int(0.25 * pw)), py2]

        person_detected_ppe = []
        has_goggles = False
        has_mask = False
        has_lab_coat = False
        has_gloves = False

        for i, item in enumerate(ppe_items):
            cname = item["class_name"]
            ibox = item["box"]

            matched = False
            if cname == "goggles" and (_box_in_zone(ibox, head_zone) or _box_in_zone(ibox, p["box"])):
                has_goggles = True
                matched = True
            elif cname == "mask" and (_box_in_zone(ibox, head_zone) or _box_in_zone(ibox, p["box"])):
                has_mask = True
                matched = True
            elif cname == "lab_coat" and (_box_in_zone(ibox, torso_zone) or _box_in_zone(ibox, p["box"])):
                has_lab_coat = True
                matched = True
            elif cname == "gloves" and (_box_in_zone(ibox, hands_zone) or _box_in_zone(ibox, p["box"])):
                has_gloves = True
                matched = True
            elif _box_in_zone(ibox, p["box"]):
                matched = True

            if matched:
                person_detected_ppe.append(item)
                claimed_ppe.add(i)

        # Build list of missing required PPE
        missing_ppe = []
        if not has_lab_coat:
            missing_ppe.append({
                "hazard_type": "Missing Lab Coat",
                "label": "NO LAB COAT",
                "zone": torso_zone,
            })
        if ENABLE_GOGGLES_DETECTION and not has_goggles:
            missing_ppe.append({
                "hazard_type": "Missing Safety Goggles",
                "label": "NO SAFETY GOGGLES",
                "zone": head_zone,
            })
        if not has_mask:
            missing_ppe.append({
                "hazard_type": "Missing Face Mask",
                "label": "NO FACE MASK",
                "zone": head_zone,
            })
        if not has_gloves:
            missing_ppe.append({
                "hazard_type": "Missing Protective Gloves",
                "label": "NO HAND GLOVES",
                "zone": hands_zone,
            })

        is_compliant = (len(missing_ppe) == 0)
        person_ident = p.get("track_id") if p.get("track_id") is not None else (idx + 1)

        analyzed_persons.append({
            "person_id": person_ident,
            "track_id": p.get("track_id"),
            "box": p["box"],
            "conf": p["conf"],
            "head_zone": head_zone,
            "torso_zone": torso_zone,
            "hands_zone": hands_zone,
            "detected_ppe": person_detected_ppe,
            "missing_ppe": missing_ppe,
            "is_compliant": is_compliant,
            "has_goggles": has_goggles if ENABLE_GOGGLES_DETECTION else True,
            "has_mask": has_mask,
            "has_lab_coat": has_lab_coat,
            "has_gloves": has_gloves,
        })

    standalone_ppe = [
        item for i, item in enumerate(ppe_items) if i not in claimed_ppe
    ]

    # periodic contraband check on workbench surface (cached between runs)
    contraband_items = get_cached_contraband(frame, imgsz=imgsz)
    has_contraband = len(contraband_items) > 0

    # fire and smoke hazard detection
    process_hazards(frame, current_frame_idx)
    hazard_state = get_hazard_state()
    fire_active = hazard_state.get("fire_active", False)
    smoke_active = hazard_state.get("smoke_active", False)
    hazard_boxes = hazard_state.get("hazards", [])

    # log active fire/smoke hazards
    now = time.time()
    if fire_active and (now - _last_hazard_log_time > 5.0):
        _last_hazard_log_time = now
        logger.critical("CRITICAL: FIRE HAZARD DETECTED in laboratory area!")
    elif smoke_active and (now - _last_hazard_log_time > 5.0):
        _last_hazard_log_time = now
        logger.warning("WARNING: SMOKE HAZARD DETECTED in laboratory area!")

    has_emergency = fire_active or smoke_active
    has_violations = any(not p["is_compliant"] for p in analyzed_persons) or has_contraband or has_emergency
    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "timestamp": time.time(),
        "persons_count": len(analyzed_persons),
        "persons": analyzed_persons,
        "standalone_ppe": standalone_ppe,
        "contraband": contraband_items,
        "contraband_count": len(contraband_items),
        "fire_active": fire_active,
        "smoke_active": smoke_active,
        "hazards": hazard_boxes,
        "has_violations": has_violations,
        "has_contraband": has_contraband,
        "has_emergency": has_emergency,
        "model_backend": backend_label,
        "backend_key": backend,
        "latency_ms": latency_ms,
        "annotated_b64": annotated_b64,
    }


def compare_backends(frame, person_conf=0.25, ppe_conf=0.15, imgsz=640, targets=None):
    """
    Run comparison across best(1), best(2), best(3), best(4), and Roboflow API on the same frame
    side-by-side to compare detection speed, person count, PPE items, and compliance.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {"error": "Invalid frame for comparison"}

    if targets is None:
        targets = ["roboflow", "cloud_api"]

    model_results = {}
    for t in targets:
        k = normalize_backend_id(t)
        if k not in AVAILABLE_MODELS:
            continue
        info = AVAILABLE_MODELS[k]
        analysis = analyze_frame_ppe(
            frame, person_conf=person_conf, ppe_conf=ppe_conf, imgsz=imgsz, backend_override=k
        )
        lat = analysis.get("latency_ms", 0.0)

        detected_ppe = [
            {
                "class": d.get("raw_class", d.get("class_name")),
                "confidence": d.get("confidence"),
                "box": d.get("box"),
            }
            for p in analysis.get("persons", [])
            for d in p.get("detected_ppe", [])
        ] + [
            {
                "class": d.get("raw_class", d.get("class_name")),
                "confidence": d.get("confidence"),
                "box": d.get("box"),
            }
            for d in analysis.get("standalone_ppe", [])
        ]

        missing_items = [
            m.get("label")
            for p in analysis.get("persons", [])
            for m in p.get("missing_ppe", [])
        ]

        violations = sum(1 for p in analysis.get("persons", []) if not p.get("is_compliant"))
        persons_cnt = analysis.get("persons_count", 0)
        compliance_pct = 100 if persons_cnt == 0 else round(((persons_cnt - violations) / persons_cnt) * 100)

        model_results[k] = {
            "id": k,
            "model_name": info["name"],
            "full_name": info["full_name"],
            "badge": info["badge"],
            "type": info["type"],
            "latency_ms": lat,
            "persons_count": persons_cnt,
            "detected_ppe": detected_ppe,
            "detected_count": len(detected_ppe),
            "violations_count": violations,
            "compliance_pct": compliance_pct,
            "missing_items": missing_items,
            "is_active": (k == get_active_backend()),
            "analysis": analysis,
        }

    # Calculate comparative metrics
    valid_lats = [(k, v["latency_ms"]) for k, v in model_results.items() if v["latency_ms"] > 0]
    fastest_k = min(valid_lats, key=lambda x: x[1])[0] if valid_lats else None
    most_dets_k = max(model_results.keys(), key=lambda k: model_results[k]["detected_count"]) if model_results else None

    # Calculate comparative speed between the two API backends
    rf_lat = model_results.get("roboflow", {}).get("latency_ms", 0.0)
    cloud_lat = model_results.get("cloud_api", {}).get("latency_ms", 0.0)

    for k, v in model_results.items():
        if k == "roboflow":
            if cloud_lat > 0 and rf_lat > 0:
                ratio = round(cloud_lat / max(rf_lat, 1.0), 2)
                v["speedup_vs_api"] = f"{ratio}x vs Cloud API" if ratio != 1.0 else "Baseline"
            else:
                v["speedup_vs_api"] = "Roboflow API"
        elif k == "cloud_api":
            if rf_lat > 0 and cloud_lat > 0:
                ratio = round(rf_lat / max(cloud_lat, 1.0), 2)
                v["speedup_vs_api"] = f"{ratio}x vs Roboflow API" if ratio != 1.0 else "Baseline"
            else:
                v["speedup_vs_api"] = "Cloud API"
        else:
            v["speedup_vs_api"] = "N/A"

    speedup_summary = f"{model_results[fastest_k]['model_name']} is fastest at {model_results[fastest_k]['latency_ms']} ms" if fastest_k else "N/A"

    return {
        "timestamp": time.time(),
        "active_backend": get_active_backend(),
        "speedup": speedup_summary,
        "fastest_model": fastest_k,
        "most_detections_model": most_dets_k,
        "models": model_results,
        "roboflow": model_results.get("roboflow", {}),
        "cloud_api": model_results.get("cloud_api", {}),
    }


def _draw_corner_rect(img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
    """Draw tactical HUD corner brackets around a bounding box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cl = min(corner_len, max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4))
    # Top-left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness + 1)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness + 1)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness + 1)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness + 1)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness + 1)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness + 1)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness + 1)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness + 1)


def render_safety_overlay(frame, analysis, camera_code="CAM-01"):
    """
    Clean, uncluttered, state-of-the-art AI overlay:
    - Dynamic top HUD displaying active model (Local YOLO best(1..4) vs Roboflow Cloud API) and latency
    - Single corner-bracketed bounding box per person
    - Single consolidated status pill above person (e.g. 'PERSON #1 | MISSING: GOGGLES, MASK')
    - Clean thin green outlines on detected PPE items
    - Eliminates redundant nested face and torso boxes for maximum clarity
    """
    if frame is None:
        return None

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    contraband_list = analysis.get("contraband", []) if analysis else []
    fire_active = analysis.get("fire_active", False) if analysis else False
    smoke_active = analysis.get("smoke_active", False) if analysis else False
    hazard_boxes = analysis.get("hazards", []) if analysis else []

    # Minimalist Translucent Top HUD (slice-blend in place: 130x faster than full-frame copy)
    bar_w = 600 if (contraband_list or fire_active or smoke_active or (analysis and "hybrid" in str(analysis.get("backend_key", "")).lower())) else 420
    bar_h = 38
    # offset hud downwards if top alert banner is showing
    hud_y_offset = 48 if (fire_active or smoke_active) else 0
    hx1, hy1, hx2, hy2 = 16, 14 + hud_y_offset, min(w, 16 + bar_w), min(h, 14 + bar_h + hud_y_offset)
    hud_slice = annotated[hy1:hy2, hx1:hx2]
    if hud_slice.shape[0] > 0 and hud_slice.shape[1] > 0:
        dark_bar = np.full_like(hud_slice, 12)
        cv2.addWeighted(dark_bar, 0.70, hud_slice, 0.30, 0, hud_slice)
    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (55, 55, 55), 1)

    persons = analysis.get("persons", []) if analysis else []
    has_violations = analysis.get("has_violations", False) if analysis else False

    # Status LED
    led_color = (0, 0, 255) if (fire_active or smoke_active) else ((40, 40, 235) if (has_violations or contraband_list) else ((40, 210, 40) if persons else (180, 180, 180)))
    cv2.circle(annotated, (32, 33 + hud_y_offset), 5, led_color, -1)

    model_b = analysis.get("model_backend", "") if analysis else ""
    backend_k = analysis.get("backend_key", get_active_backend())
    if backend_k == "hybrid" or "hybrid" in str(model_b).lower():
        backend_tag = "HYBRID (YOLO + API)"
    elif backend_k == "cloud_api" or "cloud_api" in str(model_b).lower():
        backend_tag = "ROBOFLOW CLOUD (YOLO26n)"
    elif "roboflow" in str(backend_k).lower() or "roboflow" in str(model_b).lower():
        backend_tag = "ROBOFLOW API"
    elif backend_k in AVAILABLE_MODELS:
        backend_tag = f"LOCAL YOLO {AVAILABLE_MODELS[backend_k]['name']}.pt"
    else:
        backend_tag = f"LOCAL YOLO {get_active_backend()}"

    lat = analysis.get("latency_ms") if analysis else None
    lat_str = f" | {lat:.0f}ms" if lat is not None else ""
    hud_title = f"{backend_tag} | {camera_code}{lat_str}"
    cv2.putText(annotated, hud_title, (46, 28 + hud_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

    if fire_active:
        status_txt = "CRITICAL EMERGENCY: FIRE HAZARD DETECTED"
        txt_color = (40, 40, 255)
    elif smoke_active:
        status_txt = "CRITICAL WARNING: SMOKE HAZARD DETECTED"
        txt_color = (0, 165, 255)
    elif persons:
        viol_count = sum(1 for p in persons if not p.get("is_compliant"))
        if viol_count > 0:
            status_txt = f"ACTIVE INFRACTIONS: {viol_count}"
        else:
            status_txt = "MONITORING ACTIVE: PPE COMPLIANT"
        if contraband_list:
            status_txt += f"  |  CONTRABAND: {len(contraband_list)}"
        txt_color = (60, 60, 240) if (viol_count > 0 or contraband_list) else (60, 220, 60)
    elif contraband_list:
        status_txt = f"CONTRABAND ALERT: {len(contraband_list)} ITEM(S) ON BENCH"
        txt_color = (60, 60, 240)
    else:
        status_txt = "MONITORING ACTIVE: LAB AREA SECURE"
        txt_color = (140, 150, 160)

    cv2.putText(annotated, status_txt, (46, 42 + hud_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.34, txt_color, 1, cv2.LINE_AA)

    # Render each detected person (Clean, Non-Cluttered)
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        is_compliant = p.get("is_compliant", False)
        missing = p.get("missing_ppe", [])
        detected = p.get("detected_ppe", [])

        if not is_compliant:
            box_color = (40, 40, 235)  # Crimson Red
            _draw_corner_rect(annotated, px1, py1, px2, py2, box_color, thickness=2)

            # Single unified clean header badge above person (no people counter / tracking ID)
            missing_short = [m["label"].replace("NO ", "") for m in missing]
            if len(missing_short) >= 4:
                header_lbl = "[!] NO PPE DETECTED"
            else:
                header_lbl = f"MISSING: {', '.join(missing_short)}"

            (tw, th), _ = cv2.getTextSize(header_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            pill_y1 = max(0, py1 - th - 10)
            pill_y2 = max(0, py1)
            cv2.rectangle(annotated, (px1, pill_y1), (px1 + tw + 16, pill_y2), (25, 20, 180), -1)
            cv2.rectangle(annotated, (px1, pill_y1), (px1 + tw + 16, pill_y2), (50, 40, 240), 1)
            cv2.putText(annotated, header_lbl, (px1 + 8, pill_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            box_color = (40, 210, 40)  # Emerald Green
            _draw_corner_rect(annotated, px1, py1, px2, py2, box_color, thickness=2)

            comp_lbl = "[OK] PPE COMPLIANT"
            (tw, th), _ = cv2.getTextSize(comp_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            pill_y1 = max(0, py1 - th - 10)
            pill_y2 = max(0, py1)
            cv2.rectangle(annotated, (px1, pill_y1), (px1 + tw + 16, pill_y2), (20, 140, 20), -1)
            cv2.putText(annotated, comp_lbl, (px1 + 8, pill_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw detected PPE items on this person (thin clean green boxes)
        for item in detected:
            ix1, iy1, ix2, iy2 = item["box"]
            iname = item.get("raw_class", item["class_name"]).upper()
            iconf = item.get("confidence", 0.0)
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (40, 210, 40), 1)
            ilbl = f"OK {iname} {iconf*100:.0f}%"
            (itw, ith), _ = cv2.getTextSize(ilbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tag_y1 = max(0, iy1 - ith - 6)
            cv2.rectangle(annotated, (ix1, tag_y1), (ix1 + itw + 8, iy1), (20, 130, 20), -1)
            cv2.putText(annotated, ilbl, (ix1 + 4, iy1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw standalone PPE items on bench (clean amber outline)
    standalone = analysis.get("standalone_ppe", []) if analysis else []
    for item in standalone:
        ix1, iy1, ix2, iy2 = item["box"]
        iname = item.get("raw_class", item["class_name"]).upper()
        iconf = item.get("confidence", 0.0)
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (255, 170, 0), 1)
        ilbl = f"BENCH: {iname} {iconf*100:.0f}%"
        cv2.putText(annotated, ilbl, (ix1, max(14, iy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 190, 40), 1, cv2.LINE_AA)

    # bench table boundary (hidden by default for a clean video feed; set SAFELABS_SHOW_BENCH_ROI=1 to display)
    if os.environ.get("SAFELABS_SHOW_BENCH_ROI", "0") == "1":
        roi_color = (40, 40, 235) if contraband_list else (75, 85, 95)
        cv2.polylines(annotated, [BENCH_ROI], isClosed=True, color=roi_color, thickness=1, lineType=cv2.LINE_AA)

    # draw contraband violation boxes on workbench surface
    for cb in contraband_list:
        cx1, cy1, cx2, cy2 = cb["box"]
        cname = cb.get("class_name", "item").upper()
        cconf = cb.get("confidence", 0.0)
        ax, ay = cb.get("anchor", [(cx1 + cx2) / 2.0, cy2])

        # alert box around contraband item
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (30, 40, 240), 2)
        # bottom-center anchor foot resting on bench
        cv2.circle(annotated, (int(ax), int(ay)), 4, (0, 215, 255), -1)

        clbl = f"CONTRABAND: {cname} {cconf * 100:.0f}%"
        (ctw, cth), _ = cv2.getTextSize(clbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tag_y1 = max(0, cy1 - cth - 8)
        cv2.rectangle(annotated, (cx1, tag_y1), (cx1 + ctw + 8, cy1), (20, 20, 190), -1)
        cv2.putText(annotated, clbl, (cx1 + 4, cy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # draw fire and smoke hazard boxes
    for h in hazard_boxes:
        hx1, hy1, hx2, hy2 = h["box"]
        hlbl = h.get("label", "HAZARD")
        hcolor = (0, 0, 255) if hlbl == "FIRE" else (0, 165, 255)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), hcolor, 2)
        (htw, hth), _ = cv2.getTextSize(hlbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        tag_y = max(0, hy1 - hth - 8)
        cv2.rectangle(annotated, (hx1, tag_y), (hx1 + htw + 8, hy1), hcolor, -1)
        cv2.putText(annotated, hlbl, (hx1 + 4, hy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    # alert banner overlay for active emergency
    if fire_active or smoke_active:
        msg = "CRITICAL: FIRE HAZARD DETECTED" if fire_active else "WARNING: SMOKE DETECTED"
        cv2.rectangle(annotated, (0, 0), (w, 45), (0, 0, 220), -1)
        cv2.putText(annotated, msg, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def annotate_frame_with_detections(frame, detections):
    """
    Backwards-compatible standalone detection drawer.
    """
    if frame is None:
        return None
    annotated = frame.copy()
    color_map = {
        "goggles": (255, 180, 0),
        "gloves": (0, 200, 255),
        "lab_coat": (50, 220, 50),
        "mask": (255, 100, 100),
    }
    for d in detections:
        box = d.get("box", [])
        if len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            cls_name = d.get("class_name", "ppe")
            conf = d.get("confidence", 0.0)
            color = color_map.get(cls_name.lower(), (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name.upper()} {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, max(0, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 4, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated

