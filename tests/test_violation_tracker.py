import os
import sys
import time
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from database import db, ViolationLog
from violation_tracker import (
    ViolationDebounceEngine,
    create_evidence_montage,
    GRACE_PERIOD_SECONDS,
    COMPILE_MONTAGE_SECONDS,
)

def test_montage_creation():
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    f1[:] = (30, 30, 30)
    f2[:] = (50, 50, 50)

    montage = create_evidence_montage(f1, f2, "Missing Lab Coat", "Lab A", duration_sec=30)
    assert montage is not None, "Montage should be generated"
    assert montage.shape == (480, 1280, 3), f"Expected shape (480, 1280, 3), got {montage.shape}"
    print("[PASS] Montage creation succeeded with correct dimensions (480, 1280, 3).")

def test_debounce_grace_period():
    with app.app_context():
        engine = ViolationDebounceEngine.get_instance()
        test_cam_code = "TEST-CAM-99"

        initial_count = ViolationLog.query.filter_by(camera_id=999).count()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Simulate Person #99 missing a mask for 1.0 second (below 3.0s threshold)
        uncompliant_analysis = {
            "persons": [{
                "person_id": 99,
                "is_compliant": False,
                "missing_ppe": [{
                    "hazard_type": "Missing Face Mask",
                    "label": "NO FACE MASK"
                }]
            }],
            "contraband": []
        }

        engine.process_analysis(
            camera_id=999, camera_code=test_cam_code, location="Lab A",
            current_frame=dummy_frame, analysis=uncompliant_analysis
        )
        time.sleep(0.5)

        # 2. Person #99 puts mask back on (compliant) before 3.0s grace period elapses
        compliant_analysis = {
            "persons": [{
                "person_id": 99,
                "is_compliant": True,
                "missing_ppe": []
            }],
            "contraband": []
        }

        # Process compliant frame and simulate passage of time for dropout buffer
        for _ in range(3):
            engine.process_analysis(
                camera_id=999, camera_code=test_cam_code, location="Lab A",
                current_frame=dummy_frame, analysis=compliant_analysis
            )
            time.sleep(0.4)

        # Verify: Nothing was logged for camera 999! Grace period completely discarded transient detection
        after_count = ViolationLog.query.filter_by(camera_id=999).count()
        assert after_count == initial_count, f"Grace period failed: Expected {initial_count} rows, got {after_count}"
        with engine.tracker_lock:
            key = (test_cam_code, "person_99", "ppe")
            assert key not in engine.active_incidents, "Incident should be cleared from active tracking"

        print("[PASS] Temporal debounce grace period test passed: 0 false positive logs created.")

def test_latched_violation_and_continuous_compilation():
    with app.app_context():
        engine = ViolationDebounceEngine.get_instance()
        test_cam_code = "TEST-CAM-88"

        initial_count = ViolationLog.query.filter_by(camera_id=888).count()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        uncompliant_analysis = {
            "persons": [{
                "person_id": 88,
                "is_compliant": False,
                "missing_ppe": [{
                    "hazard_type": "Missing Lab Coat",
                    "label": "NO LAB COAT"
                }]
            }],
            "contraband": []
        }

        # 1. Feed frame at t=0
        engine.process_analysis(888, test_cam_code, "Lab A", dummy_frame, uncompliant_analysis)
        key = (test_cam_code, "person_88", "ppe")

        with engine.tracker_lock:
            assert key in engine.active_incidents, "Tracking record should exist"
            # Fast-forward incident start time by 3.5s to simulate 3.5 seconds continuous
            engine.active_incidents[key].first_detected = time.time() - 3.5

        # 2. Feed frame at t=3.5s -> should latch violation in DB
        engine.process_analysis(888, test_cam_code, "Lab A", dummy_frame, uncompliant_analysis)
        time.sleep(1.0) # wait for DB async worker

        with engine.tracker_lock:
            assert engine.active_incidents[key].is_latched is True, "Violation should be latched"
            db_id = engine.active_incidents[key].db_log_id
            assert db_id is not None, "DB log ID must be assigned"

        log_row = ViolationLog.query.get(db_id)
        assert log_row is not None, "ViolationLog record must exist in DB"
        assert log_row.hazard_type == "Missing Lab Coat"
        assert log_row.duration_seconds >= 3
        print(f"[PASS] 3.0s Debounce Latch confirmed: Violation #{db_id} created in DB.")

        # 3. Fast-forward to 31 seconds continuous non-compliance
        with engine.tracker_lock:
            engine.active_incidents[key].first_detected = time.time() - 31.0
            engine.active_incidents[key].latched_at = time.time() - 28.0

        second_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        engine.process_analysis(888, test_cam_code, "Lab A", second_frame, uncompliant_analysis)
        time.sleep(1.2) # wait for montage worker

        with engine.tracker_lock:
            assert engine.active_incidents[key].has_compiled_30s is True, "Montage compilation should trigger"

        # Check DB record updated with montage and ZERO new rows created
        db.session.refresh(log_row)
        assert "montage" in log_row.image_path, f"Image path should be montage, got {log_row.image_path}"
        assert log_row.duration_seconds >= 30, f"Duration should be >=30s, got {log_row.duration_seconds}"
        after_count = ViolationLog.query.filter_by(camera_id=888).count()
        assert after_count == initial_count + 1, "Only 1 row should exist for this continuous incident"
        print(f"[PASS] 30s Compilation verified: Record #{db_id} updated with montage evidence ({log_row.duration_seconds}s).")


def test_multi_item_ppe_consolidation_single_image():
    """
    Validates that when a person is wearing a lab coat but missing both gloves and a mask:
      1. Exactly 1 violation row is logged (not 2 separate rows).
      2. Exactly 1 evidence snapshot image is saved.
      3. The hazard_type and description explicitly flag the missing gloves and mask.
    """
    with app.app_context():
        engine = ViolationDebounceEngine.get_instance()
        test_cam_code = "TEST-CAM-77"
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        initial_count = ViolationLog.query.filter_by(camera_id=777).count()

        # Person wearing coat, but missing mask and gloves
        analysis = {
            "persons": [{
                "person_id": 77,
                "is_compliant": False,
                "has_lab_coat": True,
                "has_mask": False,
                "has_gloves": False,
                "missing_ppe": [
                    {"hazard_type": "Missing Face Mask", "label": "NO FACE MASK"},
                    {"hazard_type": "Missing Protective Gloves", "label": "NO HAND GLOVES"},
                ],
                "detected_ppe": [
                    {"class_name": "lab_coat", "box": [100, 100, 200, 300], "confidence": 0.94}
                ]
            }],
            "contraband": []
        }

        # 1. Feed frame at t=0
        engine.process_analysis(777, test_cam_code, "Lab Room 4", dummy_frame, analysis)
        key = (test_cam_code, "person_77", "ppe")

        with engine.tracker_lock:
            assert key in engine.active_incidents, "Incident key should be tracked"
            incident = engine.active_incidents[key]
            assert incident.hazard_type == "Missing Face Mask, Gloves"
            assert "Face Mask, Gloves" in incident.description
            # Fast-forward 3.5s to pass grace period
            incident.first_detected = time.time() - 3.5

        # 2. Feed frame at t=3.5s -> latches single violation
        engine.process_analysis(777, test_cam_code, "Lab Room 4", dummy_frame, analysis)
        time.sleep(1.0) # wait for DB async worker

        with engine.tracker_lock:
            assert engine.active_incidents[key].is_latched is True
            db_id = engine.active_incidents[key].db_log_id
            assert db_id is not None

        new_count = ViolationLog.query.filter_by(camera_id=777).count()
        assert new_count == initial_count + 1, f"Expected exactly 1 new violation row, got {new_count - initial_count}"

        log_row = ViolationLog.query.get(db_id)
        assert log_row is not None
        assert log_row.hazard_type == "Missing Face Mask, Gloves", f"Unexpected hazard_type: {log_row.hazard_type}"
        assert "Face Mask, Gloves" in log_row.description
        assert "Person #" not in log_row.description, f"Unexpected person numbering in description: {log_row.description}"
        assert log_row.image_path is not None
        print(f"[PASS] Multi-item PPE consolidation verified: 1 row (#{db_id}) and 1 image saved for Missing Face Mask, Gloves.")


def test_spatial_handoff_and_progressive_multi_ppe_single_log():
    """
    Validates that when a single person shifts track_id (e.g. 50 -> 51 at the same box)
    and progressively triggers additional missing PPE items (Lab Coat -> Lab Coat, Face Mask, Gloves):
      1. Spatial handoff preserves the existing latched incident (exact 1 DB row, no duplicate).
      2. The existing DB row's hazard_type and description update in-place to consolidate all items.
    """
    with app.app_context():
        engine = ViolationDebounceEngine.get_instance()
        test_cam_code = "TEST-CAM-HANDOFF"
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_id = 666

        initial_count = ViolationLog.query.filter_by(camera_id=cam_id).count()

        # 1. Start at t=0 with track_id 50 missing Lab Coat
        analysis_t0 = {
            "persons": [{
                "person_id": 50,
                "box": [200, 100, 400, 460],
                "is_compliant": False,
                "missing_ppe": [
                    {"hazard_type": "Missing Lab Coat", "label": "NO LAB COAT"},
                ],
            }],
            "contraband": [],
        }
        engine.process_analysis(cam_id, test_cam_code, "Lab A", dummy_frame, analysis_t0)
        key_50 = (test_cam_code, "person_50", "ppe")

        with engine.tracker_lock:
            assert key_50 in engine.active_incidents
            engine.active_incidents[key_50].first_detected = time.time() - 3.5

        # 2. Latch at t=3.5s
        engine.process_analysis(cam_id, test_cam_code, "Lab A", dummy_frame, analysis_t0)
        time.sleep(0.8)

        with engine.tracker_lock:
            assert engine.active_incidents[key_50].is_latched is True
            db_id = engine.active_incidents[key_50].db_log_id
            assert db_id is not None

        # 3. Person ID shifts 50 -> 51 at overlapping bbox and also misses Face Mask & Gloves
        analysis_t4 = {
            "persons": [{
                "person_id": 51,
                "box": [205, 102, 405, 462],
                "is_compliant": False,
                "missing_ppe": [
                    {"hazard_type": "Missing Lab Coat", "label": "NO LAB COAT"},
                    {"hazard_type": "Missing Face Mask", "label": "NO FACE MASK"},
                    {"hazard_type": "Missing Protective Gloves", "label": "NO HAND GLOVES"},
                ],
            }],
            "contraband": [],
        }
        engine.process_analysis(cam_id, test_cam_code, "Lab A", dummy_frame, analysis_t4)
        time.sleep(0.8)

        after_count = ViolationLog.query.filter_by(camera_id=cam_id).count()
        assert after_count == initial_count + 1, f"Expected 1 consolidated DB row after handoff, got {after_count - initial_count}"

        with engine.tracker_lock:
            key_51 = (test_cam_code, "person_51", "ppe")
            assert key_50 not in engine.active_incidents, "Old key should migrate to new track_id"
            assert key_51 in engine.active_incidents
            assert engine.active_incidents[key_51].db_log_id == db_id

        db.session.expire_all()
        log_row = db.session.get(ViolationLog, db_id)
        assert log_row is not None
        assert log_row.hazard_type == "PPE Violation (Lab Coat, Face Mask, Gloves)", (
            f"Expected merged hazard_type, got {log_row.hazard_type}"
        )
        assert "Lab Coat, Face Mask, Gloves" in log_row.description
        assert "Person #" not in log_row.description, f"Unexpected person numbering in description: {log_row.description}"
        print(f"[PASS] Spatial handoff + in-place multi-PPE merge verified: 1 DB row (#{db_id}) updated to {log_row.hazard_type}.")


def test_general_lab_ppe_violation_visible_in_current_tab():
    """
    Validates that when no scheduled class is currently active (ongoing_id is None),
    a General Lab PPE violation (class_schedule_id=None) appears in GET /violations (tab='current').
    """
    from database import User, ClassSchedule
    from safety_engine import get_current_ongoing_class
    from datetime import time as dt_time

    with app.app_context():
        # Temporarily deactivate any ongoing class if running during a seeded class window
        toggled_ids = []
        ongoing = get_current_ongoing_class(db.session, ClassSchedule)
        while ongoing is not None:
            toggled_ids.append(ongoing.id)
            ongoing.is_active = False
            db.session.commit()
            ongoing = get_current_ongoing_class(db.session, ClassSchedule)

        admin = User.query.filter_by(role="admin").first()
        assert admin is not None, "Admin user required"

        test_v = ViolationLog(
            hazard_type="PPE Violation (Lab Coat, Face Mask, Gloves)",
            description="Person #901 observed without required Lab Coat, Face Mask, Gloves at Lab A.",
            severity="high",
            confidence_score=0.90,
            duration_seconds=3,
            camera_id=1,
            class_schedule_id=None,
            is_acknowledged=False,
            review_status="pending",
        )
        db.session.add(test_v)
        db.session.commit()
        test_vid = test_v.id

        try:
            with app.test_client() as client:
                with client.session_transaction() as sess:
                    sess["_user_id"] = str(admin.id)
                    sess["_fresh"] = True
                resp = client.get("/violations")
                assert resp.status_code == 200
                html = resp.get_data(as_text=True)
                assert "PPE Violation (Lab Coat, Face Mask, Gloves)" in html, "General Lab PPE violation missing from default /violations current tab"
                assert "No active infractions detected" not in html, "Default /violations current tab falsely showed empty state"
                assert "Person #" not in html, "Violation logs page must not display Person # numbering"
            print("[PASS] General Lab PPE violation visible on default /violations (current tab) when no class is ongoing.")
        finally:
            row = db.session.get(ViolationLog, test_vid)
            if row is not None:
                db.session.delete(row)
            for cid in toggled_ids:
                cs = db.session.get(ClassSchedule, cid)
                if cs is not None:
                    cs.is_active = True
            db.session.commit()


if __name__ == "__main__":
    test_montage_creation()
    test_debounce_grace_period()
    test_latched_violation_and_continuous_compilation()
    test_multi_item_ppe_consolidation_single_image()
    test_spatial_handoff_and_progressive_multi_ppe_single_log()
    test_general_lab_ppe_violation_visible_in_current_tab()
    print("\nALL VIOLATION DEBOUNCE & COMPILATION TESTS PASSED!")
