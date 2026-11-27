"""
REST API + simulation Blueprint.
Detection simulation and edge YOLOv8 ingestion endpoints.
"""
import os
from flask import Blueprint, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user

from database import (
    db, ViolationLog, ClassSchedule, EmergencyAlert,
    CameraFeed, HazardMaster,
)
from vision import get_camera
from auth_utils import ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR, role_required
from audit_utils import log_audit_event
from safety_engine import (
    get_current_ongoing_class, is_monitoring_enabled_for_hazard,
    save_evidence_frame, log_violation_event, log_emergency_event,
    generate_confidence_score, run_yolo_inference,
    annotate_frame_with_detections, get_yolo_model,
    analyze_frame_ppe, render_safety_overlay,
    get_active_backend, set_active_backend, compare_backends,
    AVAILABLE_MODELS, get_available_models, normalize_backend_id,
    get_hardware_telemetry, set_compute_device,
    get_objective2_model_catalog, benchmark_models_runtime,
)

api_bp = Blueprint('api', __name__)


# ───────────────────────── Simulation ─────────────────────────

@api_bp.route('/api/simulate', methods=['POST'])
@login_required
def api_simulate():
    """
    Trigger detection event from active camera feed.
    Hazard types are fetched directly from database (HazardMaster).
    """
    kind = request.args.get('kind', 'violation')
    cam_id = request.args.get('camera_id', type=int) or 1
    camera_record = db.session.get(CameraFeed, cam_id)
    code = camera_record.code if camera_record else "CAM-01"
    name = camera_record.name if camera_record else "Live Camera"
    loc = camera_record.location if camera_record else "Lab Area"

    cam = get_camera(
        camera_id=cam_id, code=code, name=name, location=loc
    )
    frame = cam.get_frame()

    if kind == 'alert':
        if current_user.role not in (ROLE_ADMIN, ROLE_AFM):
            flash("Only Admin and AFM can trigger emergency alerts.", 'error')
            return redirect(url_for('violations.violations_list', incident_filter='hazard'))

        alert_hazards = HazardMaster.query.filter_by(
            category='hazard', is_active=True
        ).all()
        if not alert_hazards:
            alert_hazards = HazardMaster.query.filter_by(
                is_active=True
            ).all()

        count = EmergencyAlert.query.count()
        selected = alert_hazards[count % len(alert_hazards)]
        alert_type = selected.hazard_type.lower().replace(' ', '_')

        static_dir = current_app.static_folder or os.path.join(
            current_app.root_path, 'static'
        )
        image_rel_path, evidence_hash = save_evidence_frame(
            frame, static_dir, prefix=f'alert_{alert_type}',
            hazard_type=selected.hazard_type,
            severity='critical',
            confidence_score=None,
        )

        msg = (
            f"{selected.hazard_type}: {selected.default_description} "
            f"observed at {loc}."
        )
        log_emergency_event(
            db.session, EmergencyAlert,
            alert_type=alert_type,
            message=msg,
            image_rel_path=image_rel_path,
            evidence_hash=evidence_hash,
            camera_id=cam_id,
        )

        # Dispatch Telegram emergency alert (photo + spoken voice message)
        try:
            from notifications import dispatch_hazard_alert
            dispatch_hazard_alert(
                hazard_type=selected.hazard_type,
                location=loc,
                frame_bgr=frame
            )
        except Exception as e:
            current_app.logger.error(f"Failed to dispatch Telegram hazard alert: {e}")
        flash(
            f'Emergency alert: "{selected.hazard_type}" logged with evidence!',
            'warning',
        )
        return redirect(url_for('violations.violations_list', incident_filter='hazard'))

    # Restrict manual evidence capture (violation) to Admin and AFM
    if current_user.role not in (ROLE_ADMIN, ROLE_AFM):
        flash("Access denied: Only Admin and AFM personnel can capture live evidence.", 'error')
        return redirect(url_for('video.live_feed', camera_id=cam_id))

    # Run YOLOv8 Person + PPE compliance analysis on live frame
    analysis = analyze_frame_ppe(frame, person_conf=0.25, ppe_conf=0.15, imgsz=640)
    frame_to_save = render_safety_overlay(frame, analysis, camera_code=code)

    persons = analysis.get("persons", [])
    violations_found = []

    for p in persons:
        valid_missing = [
            m for m in p.get("missing_ppe", [])
            if "goggle" not in m.get("hazard_type", "").lower()
            and "eye" not in m.get("hazard_type", "").lower()
        ]
        if not valid_missing:
            continue

        missing_items = []
        for m in valid_missing:
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

        if len(hazard_type) > 50:
            hazard_type = f"Missing {items_str}"[:50]

        severity = (
            "high"
            if any(item in ("Face Mask", "Lab Coat") for item in missing_items) or len(missing_items) > 1
            else "medium"
        )
        violations_found.append({
            "hazard_type": hazard_type,
            "person_id": p.get("person_id", "unknown"),
            "label": items_str,
            "confidence": round(float(p.get("conf", 0.85)), 3),
            "severity": severity,
            "desc": hazard_type
        })

    # Workbench contraband detections
    for cb in analysis.get("contraband", []):
        cname = cb.get("class_name", "item").capitalize()
        violations_found.append({
            "hazard_type": f"Contraband ({cname})",
            "person_id": "Workbench",
            "label": f"CONTRABAND: {cname.upper()}",
            "confidence": round(float(cb.get("confidence", 0.85)), 3),
            "severity": "high" if cname in ("Bottle", "Cup", "Pizza", "Sandwich", "Apple") else "medium",
            "desc": f"Unauthorized {cname.lower()} detected on active laboratory workbench surface."
        })

    if violations_found:
        primary = violations_found[0]
        hazard_type = primary["hazard_type"]
        desc = primary["desc"]
        severity = primary["severity"]
        confidence = primary["confidence"]
        flash_msg = f'PPE Safety Violation Detected: {hazard_type}!'
        flash_cat = 'warning'

        # Also log any secondary violations for complete audit record
        ongoing = get_current_ongoing_class(db.session, ClassSchedule)
        class_id = ongoing.id if ongoing else None
        static_dir = current_app.static_folder or os.path.join(
            current_app.root_path, 'static'
        )
        image_rel_path, evidence_hash = save_evidence_frame(
            frame_to_save, static_dir, prefix='violation',
            hazard_type=hazard_type,
            severity=severity,
            confidence_score=confidence,
        )
        for v in violations_found:
            log_violation_event(
                db.session, ViolationLog,
                hazard_type=v["hazard_type"],
                description=v["desc"],
                severity=v["severity"],
                confidence_score=v["confidence"],
                image_rel_path=image_rel_path,
                evidence_hash=evidence_hash,
                class_schedule_id=class_id,
                camera_id=cam_id,
            )
        flash(flash_msg, flash_cat)
        return redirect(url_for('violations.violations_list', tab='current'))

    elif persons:
        # All persons present are fully compliant
        hazard_type = "Full PPE Compliance Verified"
        desc = f"All {len(persons)} laboratory personnel observed with full PPE compliance."
        severity = "low"
        confidence = 0.95
        flash_msg = f'PPE Compliance Audit: All {len(persons)} observed personnel are fully compliant!'
        flash_cat = 'success'
    else:
        # Fallback to configured hazard master rules when bench is unoccupied
        ppe_hazards = HazardMaster.query.filter_by(
            category='ppe', is_active=True
        ).all()
        if not ppe_hazards:
            ppe_hazards = HazardMaster.query.filter_by(is_active=True).all()

        count = ViolationLog.query.count()
        selected_rule = ppe_hazards[count % len(ppe_hazards)]
        hazard_type = selected_rule.hazard_type
        desc = selected_rule.default_description
        severity = selected_rule.default_severity
        confidence = generate_confidence_score()
        flash_msg = f'Laboratory Scan Event: "{hazard_type}" logged with evidence frame!'
        flash_cat = 'info'

    ongoing = get_current_ongoing_class(db.session, ClassSchedule)
    class_id = ongoing.id if ongoing else None

    static_dir = current_app.static_folder or os.path.join(
        current_app.root_path, 'static'
    )
    image_rel_path, evidence_hash = save_evidence_frame(
        frame_to_save, static_dir, prefix='violation',
        hazard_type=hazard_type,
        severity=severity,
        confidence_score=confidence,
    )
    log_violation_event(
        db.session, ViolationLog,
        hazard_type=hazard_type,
        description=desc,
        severity=severity,
        confidence_score=confidence,
        image_rel_path=image_rel_path,
        evidence_hash=evidence_hash,
        class_schedule_id=class_id,
        camera_id=cam_id,
    )

    log_audit_event(
        'manual_evidence_capture',
        resource_type='camera_feed',
        resource_id=cam_id,
        details=f"Manual evidence capture on {code} ({loc}) by {current_user.username}"
    )

    flash(flash_msg, flash_cat)
    return redirect(url_for('violations.violations_list', tab='current'))


@api_bp.route('/capture_evidence', methods=['POST'])
@login_required
def capture_evidence():
    """Manual evidence capture endpoint for Admin and AFM."""
    if current_user.role not in (ROLE_ADMIN, ROLE_AFM):
        flash("Access denied: Only Admin and AFM personnel can capture live evidence.", 'error')
        return redirect(url_for('video.live_feed'))
    return api_simulate()


@api_bp.route('/api/camera/<int:camera_id>/detect', methods=['GET', 'POST'])
@login_required
def api_camera_detect(camera_id=1):
    """
    Run real-time PPE detection on the active camera stream frame using active backend.
    Returns JSON with detected PPE, bounding boxes, confidence, and model metadata.
    """
    cam_record = db.session.get(CameraFeed, camera_id)
    code = cam_record.code if cam_record else f"CAM-{camera_id:02d}"
    cam = get_camera(camera_id=camera_id)
    frame = cam.get_frame(overlay_ai=False)
    analysis = analyze_frame_ppe(frame, person_conf=0.25, ppe_conf=0.15, imgsz=640)
    model = get_yolo_model()
    class_names = list(model.names.values()) if model else []
    active_b = get_active_backend()
    model_info = AVAILABLE_MODELS.get(active_b, {})
    return {
        "status": "success",
        "camera_id": camera_id,
        "camera_code": code,
        "analysis": analysis,
        "classes": class_names,
        "active_backend": active_b,
        "model": model_info.get("full_name", active_b),
    }


@api_bp.route('/api/model/backend', methods=['GET', 'POST'])
@login_required
def api_model_backend():
    """
    Get or switch the active detection backend ('roboflow', 'cloud_api', 'hybrid').
    Zero-downtime switch without server restart.
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        target = data.get('backend')
        if not target:
            return {
                "status": "error",
                "message": "Missing backend parameter",
            }, 400
        try:
            active = set_active_backend(target)
            model_info = AVAILABLE_MODELS.get(active, {})
            return {
                "status": "success",
                "active_backend": active,
                "model_name": model_info.get("full_name", active),
                "model_short": model_info.get("name", active),
                "message": f"Successfully switched detection backend to {model_info.get('name', active)}",
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}, 400
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    active = get_active_backend()
    model_info = AVAILABLE_MODELS.get(active, {})
    return {
        "status": "success",
        "active_backend": active,
        "model_name": model_info.get("full_name", active),
        "model_short": model_info.get("name", active),
        "available_models": get_available_models(),
        "local_model_path": model_info.get("path"),
        "roboflow_workflow_id": os.environ.get("ROBOFLOW_WORKFLOW_ID", "general-segmentation-api-4"),
    }


@api_bp.route('/api/model/compare', methods=['GET', 'POST'])
@login_required
def api_model_compare():
    """
    Side-by-side benchmark comparison: runs best(1), best(2), best(3), best(4), and Roboflow API
    on the current camera frame and returns detection speed, person count, PPE items, and compliance.
    """
    cam_id = request.args.get('camera_id', type=int) or 1
    cam = get_camera(camera_id=cam_id)
    frame = cam.get_frame(overlay_ai=False)
    if frame is None:
        # fallback to sample evidence frame if camera offline
        evidence_dir = os.path.join(current_app.root_path, "static", "evidence")
        if os.path.exists(evidence_dir):
            import cv2
            for f in os.listdir(evidence_dir):
                if f.endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('.enc'):
                    frame = cv2.imread(os.path.join(evidence_dir, f))
                    if frame is not None:
                        break

    if frame is None:
        return {"status": "error", "message": "No active camera frame or sample image available for benchmark"}, 400

    models_param = request.args.get('models')
    targets = [m.strip() for m in models_param.split(',')] if models_param else None

    report = compare_backends(frame, person_conf=0.25, ppe_conf=0.15, imgsz=640, targets=targets)
    return {
        "status": "success",
        "camera_id": cam_id,
        "comparison": report,
    }


# ───────────────────────── REST API ─────────────────────────

@api_bp.route('/api/violations', methods=['POST'])
def api_log_violation():
    """REST API endpoint for edge YOLOv8 model inference workers."""
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('SAFELABS_API_KEY', 'safelabs-edge-key-2026')
    if not current_user.is_authenticated and api_key != expected_key:
        return {
            'error': 'Unauthorized: valid X-API-KEY header or session required'
        }, 401

    data = request.get_json(silent=True) or request.form
    hazard_type = data.get('hazard_type')
    if not hazard_type:
        return {'error': 'hazard_type is required'}, 400

    description = data.get('description', '')
    severity = data.get('severity', 'medium')
    confidence_score = data.get('confidence_score', type=float)
    class_id = data.get('class_id')
    camera_id = data.get('camera_id')
    schedule = None
    if class_id:
        schedule = db.session.get(ClassSchedule, int(class_id))
    else:
        schedule = get_current_ongoing_class(db.session, ClassSchedule)
        if schedule:
            class_id = schedule.id

    if schedule and not is_monitoring_enabled_for_hazard(
        schedule, hazard_type
    ):
        return {
            'status': 'ignored',
            'message': (
                f'{hazard_type} monitoring disabled for {schedule.class_name}'
            ),
        }, 200

    image_rel_path = data.get('image_path')
    if not image_rel_path:
        cam = get_camera(camera_id=int(camera_id) if camera_id else 1)
        frame = cam.get_frame()
        static_dir = current_app.static_folder or os.path.join(
            current_app.root_path, 'static'
        )
        image_rel_path, evidence_hash = save_evidence_frame(
            frame, static_dir, prefix='violation',
            hazard_type=hazard_type,
            severity=severity,
            confidence_score=confidence_score,
        )
    else:
        evidence_hash = None

    # Use provided confidence or generate a simulated one
    if confidence_score is None:
        confidence_score = generate_confidence_score()

    v = log_violation_event(
        db.session, ViolationLog,
        hazard_type=hazard_type,
        description=description,
        severity=severity,
        confidence_score=confidence_score,
        image_rel_path=image_rel_path,
        evidence_hash=evidence_hash,
        video_clip_path=data.get('video_clip_path'),
        class_schedule_id=class_id,
        camera_id=int(camera_id) if camera_id else None,
    )

    return {
        'status': 'success',
        'id': v.id,
        'hazard_type': v.hazard_type,
        'severity': v.severity,
        'confidence_score': v.confidence_score,
        'image_path': v.image_path,
        'timestamp': v.timestamp.isoformat() if v.timestamp else None,
    }, 201


@api_bp.route('/api/alerts', methods=['POST'])
def api_log_alert():
    """REST API endpoint for fire/smoke/hazard detection workers."""
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('SAFELABS_API_KEY', 'safelabs-edge-key-2026')
    if not current_user.is_authenticated and api_key != expected_key:
        return {
            'error': 'Unauthorized: valid X-API-KEY header or session required'
        }, 401

    data = request.get_json(silent=True) or request.form
    alert_type = data.get('alert_type', 'fire')
    message = data.get('message', f'Automated alert: {alert_type} detected.')
    camera_id = data.get('camera_id')

    image_rel_path = data.get('image_path')
    if not image_rel_path:
        cam = get_camera(camera_id=int(camera_id) if camera_id else 1)
        frame = cam.get_frame()
        static_dir = current_app.static_folder or os.path.join(
            current_app.root_path, 'static'
        )
        image_rel_path, evidence_hash = save_evidence_frame(
            frame, static_dir, prefix=f'alert_{alert_type}'
        )
    else:
        evidence_hash = None

    alert = log_emergency_event(
        db.session, EmergencyAlert,
        alert_type=alert_type,
        message=message,
        image_rel_path=image_rel_path,
        evidence_hash=evidence_hash,
        camera_id=int(camera_id) if camera_id else None,
    )

    return {
        'status': 'success',
        'id': alert.id,
        'alert_type': alert.alert_type,
        'message': alert.message,
        'timestamp': alert.timestamp.isoformat() if alert.timestamp else None,
    }, 201


@api_bp.route('/api/telemetry', methods=['GET'])
@login_required
def api_telemetry():
    """
    Consolidated live telemetry receiver endpoint.
    Aggregates active stream diagnostics, unacknowledged alerts & violations,
    and fire/smoke emergency signature for industry-standard Web Audio siren dispatch.
    """
    cam_id = request.args.get('camera_id', type=int) or 1
    cam_record = db.session.get(CameraFeed, cam_id)
    cam = get_camera(
        camera_id=cam_id,
        source=cam_record.stream_source if cam_record else None,
        code=cam_record.code if cam_record else f"CAM-{cam_id:02d}",
        name=cam_record.name if cam_record else f"Camera {cam_id}",
        location=cam_record.location if cam_record else "Lab Area",
    )
    telem = cam.get_telemetry() if cam else {}

    # Query unacknowledged alerts & violations
    unack_alerts_list = EmergencyAlert.query.filter_by(is_acknowledged=False).order_by(EmergencyAlert.timestamp.desc()).all()
    unack_alerts = len(unack_alerts_list)
    fire_smoke_alerts = [
        a for a in unack_alerts_list
        if any(k in (a.alert_type or '').lower() for k in ('fire', 'smoke', 'flame', 'combust'))
    ]
    fire_smoke_unack_count = len(fire_smoke_alerts)
    latest_fs_alert = fire_smoke_alerts[0] if fire_smoke_alerts else None

    unack_violations = ViolationLog.query.filter_by(is_acknowledged=False).filter(
        ViolationLog.review_status != 'false_positive'  # type: ignore[arg-type]
    ).count()

    latest_alert = unack_alerts_list[0] if unack_alerts_list else None
    latest_viol = ViolationLog.query.filter_by(is_acknowledged=False).filter(
        ViolationLog.review_status != 'false_positive'  # type: ignore[arg-type]
    ).order_by(ViolationLog.timestamp.desc()).first()

    live_fs = bool(telem.get("fire_active") or telem.get("smoke_active"))
    fire_or_smoke_active = live_fs or (fire_smoke_unack_count > 0)

    # Determine highest active severity
    sev = telem.get("severity")
    if fire_or_smoke_active:
        sev = "critical"
    elif not sev and latest_viol:
        sev = latest_viol.severity.lower()

    latest_incident_id = 0
    if latest_alert:
        latest_incident_id = max(latest_incident_id, latest_alert.id)
    if latest_viol:
        latest_incident_id = max(latest_incident_id, latest_viol.id)

    emergency_signature = None
    if live_fs:
        emergency_signature = f"live_fs_{cam_id}_{'fire' if telem.get('fire_active') else 'smoke'}"
    elif latest_fs_alert:
        emergency_signature = f"alert_fs_{latest_fs_alert.id}"

    telem.update({
        "unack_alerts": unack_alerts,
        "fire_smoke_unack_count": fire_smoke_unack_count,
        "fire_or_smoke_active": fire_or_smoke_active,
        "emergency_signature": emergency_signature,
        "unack_violations": unack_violations,
        "unacknowledged_total": unack_alerts + unack_violations,
        "highest_severity": sev,
        "latest_incident_id": latest_incident_id,
        "latest_incident_type": latest_fs_alert.alert_type if latest_fs_alert else (latest_viol.hazard_type if latest_viol else None),
    })
    return telem


@api_bp.route('/api/hardware/status', methods=['GET'])
@login_required
def api_hardware_status():
    """Return active CPU/GPU compute device, RAM RSS, CUDA VRAM, and primary camera latency."""
    cam_id = request.args.get('camera_id', type=int) or 1
    hw = get_hardware_telemetry()
    try:
        cam = get_camera(camera_id=cam_id)
        telem = cam.get_telemetry() if cam else {}
        hw["camera_latency_ms"] = telem.get("latency_ms", 0.0)
        hw["camera_fps"] = telem.get("effective_fps", 0.0)
        hw["camera_connected"] = telem.get("connected", False)
    except Exception:
        hw["camera_latency_ms"] = 0.0
        hw["camera_fps"] = 0.0
        hw["camera_connected"] = False
    return {"status": "success", "hardware": hw}


@api_bp.route('/api/hardware/device', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def api_hardware_set_device():
    """Switch active AI inference compute device live between GPU ('cuda') and CPU ('cpu')."""
    data = request.get_json(silent=True) or request.form
    target = data.get('device')
    if not target:
        return {"status": "error", "message": "Missing device parameter ('cuda' or 'cpu')."}, 400
    try:
        hw = set_compute_device(target)
        log_audit_event(
            'compute_device_switch',
            resource_type='hardware',
            details=f"Switched live AI inference compute device to {hw['active_device'].upper()} ({hw['active_device_label']})",
        )
        return {"status": "success", "hardware": hw}
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@api_bp.route('/api/models/evaluation', methods=['GET'])
@login_required
def api_models_evaluation():
    """Return Objective 2 model metrics, checkpoint comparison, and live hardware telemetry."""
    return {"status": "success", "catalog": get_objective2_model_catalog()}


@api_bp.route('/api/models/benchmark', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def api_models_benchmark():
    """Run live GPU/CPU latency & FPS benchmark for one or all models in Detection Rules."""
    data = request.get_json(silent=True) or request.form
    model_key = str(data.get('model_key', 'all')).strip() or 'all'
    device_mode = str(data.get('device', 'both')).strip() or 'both'
    runs = max(2, min(8, int(data.get('runs', 4) or 4)))
    try:
        result = benchmark_models_runtime(model_key=model_key, device_mode=device_mode, runs=runs)
        log_audit_event(
            'objective2_model_benchmark',
            resource_type='model_evaluation',
            details=f"Executed live Objective 2 benchmark (target={model_key}, device={device_mode}, net_rss_delta={result.get('memory_delta_mb', 0.0)}MB)",
        )
        return {"status": "success", "catalog": result}
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
