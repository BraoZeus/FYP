"""
SAFELABS — Temporal Debounce & Continuous Violation Lifecycle Tracker.

Features:
1. Temporal Debounce (The Grace Period):
   - Requires 3.0 seconds of continuous violation before flagging/logging.
   - Transient occlusions (<3.0s, e.g. head turns, chin scratches) are silently discarded.
2. Contraband Debounce:
   - Prohibited items on the workbench surface ROI require 3.0s continuous presence before logging.
3. 30-Second Continuous Incident Compiling:
   - If an infraction persists for 30s+, compiles initial and ongoing frames into a
     side-by-side evidence montage (montage.jpg) instead of saving individual spam files.
   - Updates the existing database record's duration and evidence hash (Zero duplicate rows).
4. Compliance Resolution:
   - When the PPE is re-worn or contraband removed, marks the record resolved with final duration.
"""

import os
import time
import threading
import logging
from typing import Optional, Dict, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("SAFELABS.TRACKER")

GRACE_PERIOD_SECONDS = 3.0       # 3 seconds continuous non-compliance required
COMPILE_MONTAGE_SECONDS = 30.0   # 30 seconds threshold for compiling evidence collage
PERSISTENCE_SYNC_INTERVAL = 15.0 # Frequency to update duration on ongoing incidents


def create_evidence_montage(
    panel_left: np.ndarray,
    panel_right: np.ndarray,
    hazard_type: str,
    location: str,
    duration_sec: int = 30,
) -> np.ndarray:
    """
    Stitch initial violation frame (t=3s) and ongoing frame (t=30s+)
    side-by-side into a single high-fidelity evidence montage image.
    """
    h, w = 480, 640
    p1 = cv2.resize(panel_left, (w, h))
    p2 = cv2.resize(panel_right, (w, h))

    banner_h = 36
    # Left banner (Initial latch)
    cv2.rectangle(p1, (0, 0), (w, banner_h), (18, 18, 20), -1)
    cv2.putText(
        p1,
        "INITIAL LATCH (3s Continuous)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )

    # Right banner (30s+ persistence)
    cv2.rectangle(p2, (0, 0), (w, banner_h), (18, 18, 20), -1)
    cv2.putText(
        p2,
        f"PERSISTENT INFRACTION ({duration_sec}s Continuous)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 69, 255),
        1,
        cv2.LINE_AA,
    )

    # Thin vertical dividing strip
    montage = np.hstack([p1, p2])
    cv2.line(montage, (w, 0), (w, h), (40, 40, 45), 2)
    return montage


class TrackedIncident:
    def __init__(
        self,
        key: tuple,
        camera_id: int,
        camera_code: str,
        location: str,
        hazard_type: str,
        severity: str,
        description: str,
        initial_frame: np.ndarray,
    ):
        self.key = key
        self.camera_id = camera_id
        self.camera_code = camera_code
        self.location = location
        self.hazard_type = hazard_type
        self.severity = severity
        self.description = description

        self.first_detected = time.time()
        self.last_seen = time.time()
        self.is_latched = False
        self.latched_at = 0.0

        self.db_log_id: Optional[int] = None
        self.initial_frame: Optional[np.ndarray] = initial_frame.copy() if initial_frame is not None else None
        self.initial_image_rel_path: Optional[str] = None
        self.has_compiled_30s = False
        self.last_db_sync = 0.0


class ViolationDebounceEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.active_incidents = {}  # key -> TrackedIncident
        self.tracker_lock = threading.Lock()

    def process_analysis(
        self,
        camera_id: int,
        camera_code: str,
        location: str,
        current_frame: np.ndarray,
        analysis: dict,
    ):
        """
        Evaluate frame detections against the debounce state machine.
        Invoked from the camera inference loop on every analyzed frame.
        """
        if not analysis or current_frame is None:
            return

        now = time.time()
        active_keys_this_frame = set()

        # 1. Collect uncompliant PPE infractions per tracked person (consolidated into 1 incident per person)
        persons = analysis.get("persons", [])
        for p in persons:
            if not p.get("is_compliant", True):
                person_id = p.get("person_id", "unknown")
                missing_ppe = [
                    m for m in p.get("missing_ppe", [])
                    if "goggle" not in m.get("hazard_type", "").lower()
                    and "eye" not in m.get("hazard_type", "").lower()
                ]
                if not missing_ppe:
                    continue

                missing_items = []
                for m in missing_ppe:
                    ht = m.get("hazard_type", "")
                    lbl = m.get("label", "")
                    if "coat" in ht.lower() or "coat" in lbl.lower():
                        if "Lab Coat" not in missing_items:
                            missing_items.append("Lab Coat")
                    elif "mask" in ht.lower() or "mask" in lbl.lower():
                        if "Face Mask" not in missing_items:
                            missing_items.append("Face Mask")
                    elif "glove" in ht.lower() or "hand" in lbl.lower():
                        if "Gloves" not in missing_items:
                            missing_items.append("Gloves")
                    else:
                        clean_lbl = lbl.replace("NO ", "").title()
                        if clean_lbl not in missing_items:
                            missing_items.append(clean_lbl)

                if not missing_items:
                    continue

                items_str = ", ".join(missing_items)
                if len(missing_items) == 1:
                    hazard_type = f"Missing {missing_items[0]}"
                elif len(missing_items) >= 3:
                    hazard_type = f"PPE Violation ({items_str})"
                else:
                    hazard_type = f"Missing {items_str}"

                # Enforce database column size constraint db.String(50)
                if len(hazard_type) > 50:
                    hazard_type = f"Missing {items_str}"[:50]

                severity = (
                    "high"
                    if any(item in ("Face Mask", "Lab Coat") for item in missing_items) or len(missing_items) > 1
                    else "medium"
                )
                desc = f"Person #{person_id} observed without required {items_str} at {location}."
                key = (camera_code, f"person_{person_id}", "ppe")
                active_keys_this_frame.add((key, severity, desc, hazard_type))

        # 2. Collect contraband items on workbench surface ROI
        contraband = analysis.get("contraband", [])
        for cb in contraband:
            cname = cb.get("class_name", "item").capitalize()
            hazard_type = f"Contraband ({cname})"
            severity = "high" if cname in ("Bottle", "Cup", "Pizza", "Sandwich", "Apple") else "medium"
            desc = f"Unauthorized contraband item ({cname}) on chemical workbench surface at {location}."
            key = (camera_code, f"bench_{cname.lower()}", hazard_type)
            active_keys_this_frame.add((key, severity, desc, hazard_type))

        with self.tracker_lock:
            current_keys_set = set()

            # Process all currently uncompliant entities
            for (key, severity, desc, hazard_type) in active_keys_this_frame:
                current_keys_set.add(key)

                if key not in self.active_incidents:
                    # Start debounce grace period timer (DO NOT LOG YET)
                    incident = TrackedIncident(
                        key=key,
                        camera_id=camera_id,
                        camera_code=camera_code,
                        location=location,
                        hazard_type=hazard_type,
                        severity=severity,
                        description=desc,
                        initial_frame=current_frame,
                    )
                    self.active_incidents[key] = incident
                else:
                    incident = self.active_incidents[key]
                    incident.last_seen = now
                    if not incident.is_latched:
                        incident.hazard_type = hazard_type
                        incident.description = desc
                        incident.severity = severity
                    continuous_elapsed = now - incident.first_detected

                    # Condition A: Grace period passed (>= 3.0s continuous) -> LATCH VIOLATION
                    if not incident.is_latched and continuous_elapsed >= GRACE_PERIOD_SECONDS:
                        incident.is_latched = True
                        incident.latched_at = now
                        self._latch_initial_violation(incident, current_frame)

                    # Condition B: Continuous infraction reached 30s -> COMPILE MONTAGE
                    elif (
                        incident.is_latched
                        and not incident.has_compiled_30s
                        and (now - incident.latched_at >= (COMPILE_MONTAGE_SECONDS - GRACE_PERIOD_SECONDS))
                    ):
                        incident.has_compiled_30s = True
                        incident.last_db_sync = now
                        self._compile_30s_montage(incident, current_frame)

                    # Condition C: Ongoing persistence beyond 30s -> Periodic duration sync (No new rows)
                    elif (
                        incident.is_latched
                        and (now - incident.last_db_sync >= PERSISTENCE_SYNC_INTERVAL)
                    ):
                        incident.last_db_sync = now
                        self._update_ongoing_duration(incident)

            # Check for resolved or transiently cleared infractions
            all_keys = list(self.active_incidents.keys())
            for key in all_keys:
                if key not in current_keys_set:
                    incident = self.active_incidents[key]
                    # Wait 1.0s to absorb transient 1-frame dropout flickers
                    if now - incident.last_seen >= 1.0:
                        if not incident.is_latched:
                            # Grace period worked: cleared before 3s -> SILENTLY DISCARD
                            del self.active_incidents[key]
                        else:
                            # Violation was latched and has now ended -> MARK RESOLVED
                            self._mark_resolved(incident)
                            del self.active_incidents[key]

    def _latch_initial_violation(self, incident: TrackedIncident, current_frame: np.ndarray):
        """Record the initial confirmed violation in DB after 3.0s grace period."""
        def _db_worker():
            try:
                from app import app
                from database import db, ViolationLog, ClassSchedule
                from safety_engine import save_evidence_frame, get_current_ongoing_class

                with app.app_context():
                    static_dir = app.static_folder or os.path.join(app.root_path, "static")
                    ongoing_class = get_current_ongoing_class(db.session, ClassSchedule)
                    class_id = ongoing_class.id if ongoing_class else None

                    # Save initial snapshot
                    rel_path, ev_hash = save_evidence_frame(
                        current_frame,
                        static_dir,
                        prefix=f"viol_cam{incident.camera_id}",
                        hazard_type=incident.hazard_type,
                        severity=incident.severity,
                    )
                    incident.initial_image_rel_path = rel_path

                    v = ViolationLog(
                        hazard_type=incident.hazard_type,
                        description=f"{incident.description} (Continuous: 3s)",
                        severity=incident.severity,
                        confidence_score=0.90,
                        duration_seconds=int(GRACE_PERIOD_SECONDS),
                        image_path=rel_path,
                        evidence_hash=ev_hash,
                        camera_id=incident.camera_id,
                        class_schedule_id=class_id,
                    )
                    db.session.add(v)
                    db.session.commit()
                    incident.db_log_id = v.id
                    logger.info(
                        f"[{incident.camera_code}] Latched violation #{v.id}: {incident.hazard_type} "
                        f"after {GRACE_PERIOD_SECONDS}s continuous detection."
                    )
            except Exception as e:
                logger.error(f"[{incident.camera_code}] Failed latching violation: {e}")

        threading.Thread(target=_db_worker, daemon=True).start()

    def _compile_30s_montage(self, incident: TrackedIncident, current_frame: np.ndarray):
        """Compile initial (3s) and ongoing (30s) frames into a single montage image."""
        def _compile_worker():
            try:
                from app import app
                from database import db, ViolationLog
                from safety_engine import save_evidence_frame

                if incident.initial_frame is None or not incident.db_log_id:
                    return

                with app.app_context():
                    static_dir = app.static_folder or os.path.join(app.root_path, "static")
                    elapsed_total = int(time.time() - incident.first_detected)

                    # Build side-by-side evidence montage
                    montage = create_evidence_montage(
                        panel_left=incident.initial_frame,
                        panel_right=current_frame,
                        hazard_type=incident.hazard_type,
                        location=incident.location,
                        duration_sec=elapsed_total,
                    )

                    montage_rel_path, montage_hash = save_evidence_frame(
                        montage,
                        static_dir,
                        prefix=f"montage_cam{incident.camera_id}",
                        hazard_type=incident.hazard_type,
                        severity=incident.severity,
                    )

                    record = ViolationLog.query.get(incident.db_log_id)
                    if record:
                        old_rel_path = record.image_path
                        record.image_path = montage_rel_path
                        record.evidence_hash = montage_hash
                        record.duration_seconds = elapsed_total
                        record.description = (
                            f"{incident.description} | Persistent non-compliance ({elapsed_total}s continuous compiled montage)."
                        )
                        db.session.commit()

                        # Reclaim disk space by safely removing single-panel pre-montage image
                        if old_rel_path and old_rel_path != montage_rel_path:
                            try:
                                old_full = os.path.join(static_dir, os.path.basename(old_rel_path))
                                if os.path.exists(old_full):
                                    os.remove(old_full)
                            except Exception:
                                pass

                    logger.info(
                        f"[{incident.camera_code}] Compiled 30s evidence montage for Violation #{incident.db_log_id} "
                        f"({elapsed_total}s continuous)."
                    )
            except Exception as e:
                logger.error(f"[{incident.camera_code}] Failed compiling 30s montage: {e}")

        threading.Thread(target=_compile_worker, daemon=True).start()

    def _update_ongoing_duration(self, incident: TrackedIncident):
        """Update duration on the existing record for persistent non-compliance."""
        def _update_worker():
            try:
                from app import app
                from database import db, ViolationLog

                if not incident.db_log_id:
                    return

                with app.app_context():
                    record = ViolationLog.query.get(incident.db_log_id)
                    if record:
                        elapsed_total = int(time.time() - incident.first_detected)
                        record.duration_seconds = elapsed_total
                        record.description = (
                            f"{incident.description} | Ongoing continuous non-compliance: {elapsed_total}s."
                        )
                        db.session.commit()
            except Exception as e:
                logger.debug(f"Duration sync error: {e}")

        threading.Thread(target=_update_worker, daemon=True).start()

    def _mark_resolved(self, incident: TrackedIncident):
        """Mark violation resolved when compliance is restored or entity leaves."""
        def _resolve_worker():
            try:
                from app import app
                from database import db, ViolationLog

                if not incident.db_log_id:
                    return

                with app.app_context():
                    record = ViolationLog.query.get(incident.db_log_id)
                    if record:
                        total_duration = int(incident.last_seen - incident.first_detected)
                        record.duration_seconds = max(total_duration, int(GRACE_PERIOD_SECONDS))
                        record.resolution_note = (
                            f"Compliance restored after {record.duration_seconds}s of continuous non-compliance."
                        )
                        db.session.commit()
                        logger.info(
                            f"[{incident.camera_code}] Violation #{incident.db_log_id} marked resolved: "
                            f"{record.resolution_note}"
                        )
            except Exception as e:
                logger.debug(f"Resolve update error: {e}")

        threading.Thread(target=_resolve_worker, daemon=True).start()
