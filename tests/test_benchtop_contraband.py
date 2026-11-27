"""
Unit & Integration Test: Benchtop Contraband Detection & Class Confidence Gating
Validates inclusion of laptops (COCO 63) and cell phones (COCO 67) along with
class-specific confidence gates:
  - 63 (Laptop): 0.35
  - 67 (Cell Phone): 0.28
  - default: 0.32
"""
import os
import sys
import time
import numpy as np
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from safety_engine import (
    CONTRABAND_CLASSES,
    CLASS_CONFIDENCE_THRESHOLDS,
    detect_contraband,
    get_cached_contraband,
    BENCH_ROI
)
from violation_tracker import ViolationDebounceEngine
from app import app
from database import db, ViolationLog


def test_contraband_classes_and_thresholds():
    # 1. Verify COCO class IDs are present
    assert 63 in CONTRABAND_CLASSES, "COCO class 63 (laptop) must be in CONTRABAND_CLASSES"
    assert CONTRABAND_CLASSES[63] == "laptop", f"Expected 'laptop', got {CONTRABAND_CLASSES[63]}"

    assert 67 in CONTRABAND_CLASSES, "COCO class 67 (cell phone) must be in CONTRABAND_CLASSES"
    assert CONTRABAND_CLASSES[67] == "cell phone", f"Expected 'cell phone', got {CONTRABAND_CLASSES[67]}"

    # 2. Verify confidence threshold specification
    assert CLASS_CONFIDENCE_THRESHOLDS.get(63) == 0.35, "Laptop threshold must be 0.35"
    assert CLASS_CONFIDENCE_THRESHOLDS.get(67) == 0.28, "Cell phone threshold must be 0.28"
    assert CLASS_CONFIDENCE_THRESHOLDS.get("default") == 0.32, "Default threshold must be 0.32"

    print("[PASS] CONTRABAND_CLASSES and CLASS_CONFIDENCE_THRESHOLDS verified.")


def test_confidence_gate_logic():
    # Test gate evaluation logic against mock boxes
    def evaluate_gate(cls_id, conf):
        req = CLASS_CONFIDENCE_THRESHOLDS.get(cls_id, CLASS_CONFIDENCE_THRESHOLDS.get("default", 0.32))
        return conf >= req

    # Laptop (63): 0.35 threshold
    assert not evaluate_gate(63, 0.34), "Laptop at 0.34 should be rejected"
    assert evaluate_gate(63, 0.35), "Laptop at 0.35 should be accepted"
    assert evaluate_gate(63, 0.50), "Laptop at 0.50 should be accepted"

    # Cell phone (67): 0.28 threshold
    assert not evaluate_gate(67, 0.27), "Cell phone at 0.27 should be rejected"
    assert evaluate_gate(67, 0.28), "Cell phone at 0.28 should be accepted"
    assert evaluate_gate(67, 0.30), "Cell phone at 0.30 should be accepted"

    # Default items (e.g. 39: bottle): 0.32 threshold
    assert not evaluate_gate(39, 0.31), "Bottle at 0.31 should be rejected"
    assert evaluate_gate(39, 0.32), "Bottle at 0.32 should be accepted"
    assert evaluate_gate(39, 0.40), "Bottle at 0.40 should be accepted"

    print("[PASS] Class-specific confidence gate thresholds correctly asserted.")


def test_benchtop_electronics_debounce_tracking():
    with app.app_context():
        engine = ViolationDebounceEngine.get_instance()
        test_cam_code = "TEST-BENCH-CAM"
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        initial_count = ViolationLog.query.filter_by(camera_id=777).count()

        # Simulate laptop & cell phone detected on workbench
        laptop_analysis = {
            "persons": [],
            "contraband": [
                {
                    "class_id": 63,
                    "class_name": "laptop",
                    "confidence": 0.45,
                    "box": [400, 450, 550, 550],
                    "anchor": [475.0, 550.0],
                    "on_bench": True,
                },
                {
                    "class_id": 67,
                    "class_name": "cell phone",
                    "confidence": 0.32,
                    "box": [600, 480, 650, 530],
                    "anchor": [625.0, 530.0],
                    "on_bench": True,
                }
            ]
        }

        # 1. Feed frame at t=0
        engine.process_analysis(777, test_cam_code, "Lab B", dummy_frame, laptop_analysis)
        laptop_key = (test_cam_code, "bench_laptop", "Contraband (Laptop)")
        phone_key = (test_cam_code, "bench_cell phone", "Contraband (Cell phone)")

        with engine.tracker_lock:
            assert laptop_key in engine.active_incidents, "Laptop incident should be registered"
            assert phone_key in engine.active_incidents, "Phone incident should be registered"

            # Fast-forward past 3.0s grace period (simulate 3.5s elapsed)
            engine.active_incidents[laptop_key].first_detected = time.time() - 3.5
            engine.active_incidents[phone_key].first_detected = time.time() - 3.5

        # 2. Feed second frame at t=3.5s -> latches violation in database
        engine.process_analysis(777, test_cam_code, "Lab B", dummy_frame, laptop_analysis)
        time.sleep(1.0) # wait for DB async worker

        with engine.tracker_lock:
            assert engine.active_incidents[laptop_key].is_latched is True
            assert engine.active_incidents[phone_key].is_latched is True

        new_count = ViolationLog.query.filter_by(camera_id=777).count()
        assert new_count == initial_count + 2, f"Expected 2 new violation logs, got {new_count - initial_count}"

        laptop_log = ViolationLog.query.filter_by(camera_id=777, hazard_type="Contraband (Laptop)").order_by(ViolationLog.id.desc()).first()
        phone_log = ViolationLog.query.filter_by(camera_id=777, hazard_type="Contraband (Cell phone)").order_by(ViolationLog.id.desc()).first()

        assert laptop_log is not None, "Laptop violation log must exist"
        assert phone_log is not None, "Cell phone violation log must exist"
        assert "laptop" in laptop_log.description.lower()
        assert "cell phone" in phone_log.description.lower()

        print(f"[PASS] Benchtop electronics debounced and latched: Violations #{laptop_log.id} (Laptop) and #{phone_log.id} (Cell phone).")


if __name__ == "__main__":
    test_contraband_classes_and_thresholds()
    test_confidence_gate_logic()
    test_benchtop_electronics_debounce_tracking()
    print("\nALL BENCHTOP CONTRABAND & CONFIDENCE GATE TESTS PASSED!")

