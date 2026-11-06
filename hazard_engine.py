import os
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "cpu"

model_path = os.environ.get("SAFELABS_FIRE_MODEL_PATH") or os.path.join(
    os.path.dirname(__file__), "models", "fire_smoke.pt"
)
fire_model = YOLO(model_path) if os.path.exists(model_path) else None

# Thresholds: Fire at 0.40, Smoke raised to 0.55 to prevent false alerts on diffuse lights/glare
FIRE_CONF_THRESHOLD = float(os.environ.get("SAFELABS_FIRE_CONF", "0.40"))
SMOKE_CONF_THRESHOLD = float(os.environ.get("SAFELABS_SMOKE_CONF", "0.55"))

smoke_buf = deque(maxlen=10)
fire_buf = deque(maxlen=10)
cached_hazards = []
fire_active = False
smoke_active = False


def is_overexposed_light_source(roi_bgr: np.ndarray) -> bool:
    """
    Check if a candidate detection ROI is dominated by an electric light source
    (e.g., ceiling fluorescent lamp, LED tube, bright light fixture, or specular glare).
    
    Electric lights are characterized by:
    1. Extreme luminance saturation (pixels with value >= 238).
    2. Blown-out central core with minimal texture.
    Smoke plumes are diffuse with moderate luminance and high texture variance.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    h, w = roi_bgr.shape[:2]
    if h < 4 or w < 4:
        return False

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Total box overexposure ratio
    overexposed_ratio = np.count_nonzero(gray >= 238) / float(gray.size)
    if overexposed_ratio > 0.45:
        return True

    # 2. Central core check (inner 50% region where light filaments/bulbs are hottest)
    ch1, ch2 = int(h * 0.25), int(h * 0.75)
    cw1, cw2 = int(w * 0.25), int(w * 0.75)
    core = gray[ch1:ch2, cw1:cw2]
    if core.size > 0:
        core_overexposed = np.count_nonzero(core >= 240) / float(core.size)
        if core_overexposed > 0.60:
            return True

    return False


def get_hazard_state():
    formatted = []
    for cls_id, coords, conf in cached_hazards:
        formatted.append({
            "class_id": int(cls_id),
            "label": "FIRE" if cls_id == 1 else "SMOKE",
            "box": [int(v) for v in coords],
            "confidence": round(float(conf), 3),
        })
    return {
        "fire_active": fire_active,
        "smoke_active": smoke_active,
        "hazards": formatted,
    }


def process_hazards(frame, frame_idx, return_meta=False):
    global cached_hazards, fire_active, smoke_active

    if fire_model is None or frame is None:
        if return_meta:
            return frame, get_hazard_state()
        return frame

    # sample every 5 frames for faster hazard reactivity
    if frame_idx % 5 == 0:
        min_conf = min(FIRE_CONF_THRESHOLD, SMOKE_CONF_THRESHOLD)
        pred_res = fire_model.predict(frame, conf=min_conf, device=device, verbose=False)
        res_list = list(pred_res) if pred_res is not None else []
        results = res_list[0] if len(res_list) > 0 else None
        cached_hazards = []
        has_smoke, has_fire = False, False
        h_frame, w_frame = frame.shape[:2]

        boxes = getattr(results, "boxes", None) if results is not None else None
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                # Ensure coordinates are within image boundaries
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(w_frame, x2), min(h_frame, y2)
                roi = frame[ry1:ry2, rx1:rx2]

                if cls_id == 0:  # SMOKE
                    # 1. Enforce higher confidence threshold for smoke
                    if conf < SMOKE_CONF_THRESHOLD:
                        continue

                    # 2. Ceiling fixture guard: ignore detections in top 12% if mean brightness is elevated
                    if y1 < int(h_frame * 0.12) and roi.size > 0 and roi.mean() > 210:
                        continue

                    # 3. Reject overexposed electric lamps, LED bulbs, and specular flares
                    if is_overexposed_light_source(roi):
                        continue

                    has_smoke = True
                    cached_hazards.append((cls_id, coords, conf))

                elif cls_id == 1:  # FIRE
                    if conf < FIRE_CONF_THRESHOLD:
                        continue
                    has_fire = True
                    cached_hazards.append((cls_id, coords, conf))

        smoke_buf.append(1 if has_smoke else 0)
        fire_buf.append(1 if has_fire else 0)

    # Temporal persistence threshold:
    # Smoke: requires at least 5 positive samples out of 10 (~1.0s continuous sustained detection)
    # Fire: 2 positive samples (~0.4s fast emergency latch)
    smoke_active = sum(smoke_buf) >= 5
    fire_active = sum(fire_buf) >= 2

    # draw cached detections
    for cls_id, (x1, y1, x2, y2), conf in cached_hazards:
        color = (0, 0, 255) if cls_id == 1 else (0, 165, 255)
        label = f"FIRE {conf*100:.0f}%" if cls_id == 1 else f"SMOKE {conf*100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # alert banner overlay
    if fire_active or smoke_active:
        msg = "CRITICAL: FIRE HAZARD DETECTED" if fire_active else "WARNING: SMOKE DETECTED"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 220), -1)
        cv2.putText(frame, msg, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if return_meta:
        return frame, get_hazard_state()
    return frame
