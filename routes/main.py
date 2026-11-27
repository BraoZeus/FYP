"""
Main dashboard Blueprint.
"""
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_required, current_user

from database import (
    db, ViolationLog, ClassSchedule, CameraFeed, EmergencyAlert,
)
from vision import get_camera
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR
from time_utils import get_myt_now

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@login_required
def index():
    """Index entrypoint with role-aware redirection."""
    if current_user.role == ROLE_ACADEMIC_COORDINATOR:
        return redirect(url_for('classes.classes_list'))
    return redirect(url_for('main.dashboard'))


@main_bp.route('/dashboard')
@role_required(ROLE_ADMIN, ROLE_AFM)
def dashboard():
    """Main operations overview, system statistics, and quick stream."""
    recent_violations = ViolationLog.query.filter(
        ViolationLog.review_status != 'false_positive'  # type: ignore[arg-type]
    ).order_by(
        ViolationLog.timestamp.desc()
    ).limit(5).all()

    total_violations = ViolationLog.query.filter(
        ViolationLog.review_status != 'false_positive'  # type: ignore[arg-type]
    ).count()
    unacknowledged = ViolationLog.query.filter_by(
        is_acknowledged=False
    ).filter(
        ViolationLog.review_status != 'false_positive'  # type: ignore[arg-type]
    ).count()
    active_classes = ClassSchedule.query.filter_by(is_active=True).count()

    now = get_myt_now()
    current_day = now.strftime('%A')
    current_time = now.time()
    ongoing_classes = ClassSchedule.query.filter_by(
        is_active=True
    ).filter(
        ClassSchedule.day_of_week == current_day,
        ClassSchedule.start_time <= current_time,
        ClassSchedule.end_time >= current_time,
    ).all()

    upcoming_classes = ClassSchedule.query.filter_by(
        is_active=True
    ).filter(
        ClassSchedule.day_of_week == current_day,
        ClassSchedule.start_time > current_time,
    ).order_by(ClassSchedule.start_time).all()

    recent_alerts = EmergencyAlert.query.filter_by(
        is_acknowledged=False
    ).order_by(EmergencyAlert.timestamp.desc()).limit(5).all()

    cameras = CameraFeed.query.filter_by(is_active=True).all()
    cam_id = request.args.get('camera_id', type=int) or (
        cameras[0].id if cameras else 1
    )

    selected_cam = db.session.get(CameraFeed, cam_id) if cam_id else None
    selected_cam_connected = False
    if selected_cam:
        cam_obj = get_camera(
            camera_id=selected_cam.id,
            source=selected_cam.stream_source,
            code=selected_cam.code,
            name=selected_cam.name,
            location=selected_cam.location,
        )
        selected_cam_connected = cam_obj.is_camera_connected()

    return render_template(
        'dashboard.html',
        recent_violations=recent_violations,
        total_violations=total_violations,
        unacknowledged=unacknowledged,
        active_classes=active_classes,
        ongoing_classes=ongoing_classes,
        upcoming_classes=upcoming_classes,
        recent_alerts=recent_alerts,
        cameras=cameras,
        selected_camera_id=cam_id,
        selected_cam_connected=selected_cam_connected,
    )
