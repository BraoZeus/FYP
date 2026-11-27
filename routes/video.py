"""
Video feed Blueprint — MJPEG streaming and live feed page.
"""
from flask import Blueprint, Response, render_template, request, redirect, url_for
from flask_login import login_required

from database import db, CameraFeed
from vision import generate_frames, get_camera
from safety_engine import get_available_models, get_active_backend
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM

video_bp = Blueprint('video', __name__)


@video_bp.route('/video_feed')
@video_bp.route('/video_feed/<int:camera_id>')
@role_required(ROLE_ADMIN, ROLE_AFM)
def video_feed(camera_id=1):
    """Dynamic multipart MJPEG stream for the requested camera feed (Admin & AFM only)."""
    cam_record = db.session.get(CameraFeed, camera_id)
    code = cam_record.code if cam_record else f"CAM-{camera_id:02d}"
    name = cam_record.name if cam_record else f"Camera {camera_id}"
    location = cam_record.location if cam_record else "Lab Area"
    source = cam_record.stream_source if cam_record else None

    resp = Response(
        generate_frames(
            camera_id=camera_id, source=source, code=code,
            name=name, location=location
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Connection'] = 'close'
    return resp


@video_bp.route('/live_feeds')
@video_bp.route('/live_feed')
@video_bp.route('/feed')
@role_required(ROLE_ADMIN, ROLE_AFM)
def live_feed():
    """Live surveillance control room with dropdown and Quad-view (Admin & AFM only)."""
    cameras = CameraFeed.query.filter_by(is_active=True).all()
    cam_id = request.args.get('camera_id', type=int)
    if not cam_id and cameras:
        cam_id = cameras[0].id

    selected_cam = None
    if cam_id:
        selected_cam = db.session.get(CameraFeed, cam_id)
    if not selected_cam and cameras:
        selected_cam = cameras[0]

    view_mode = request.args.get('view', 'single')  # 'single' or 'quad'

    return render_template(
        'live_feed.html',
        cameras=cameras,
        selected_camera=selected_cam,
        view_mode=view_mode,
        available_models=get_available_models(),
        active_backend=get_active_backend(),
    )


@video_bp.route('/api/camera/<int:camera_id>/status')
@role_required(ROLE_ADMIN, ROLE_AFM)
def camera_status(camera_id=1):
    """Return live telemetry diagnostics for the requested camera stream."""
    cam_record = db.session.get(CameraFeed, camera_id)
    code = cam_record.code if cam_record else f"CAM-{camera_id:02d}"
    name = cam_record.name if cam_record else f"Camera {camera_id}"
    location = cam_record.location if cam_record else "Lab Area"
    source = cam_record.stream_source if cam_record else None

    cam = get_camera(
        camera_id=camera_id, source=source, code=code,
        name=name, location=location
    )
    return cam.get_telemetry()


@video_bp.route('/capture_evidence', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def capture_evidence():
    """Manual evidence snapshot trigger for Admin and AFM."""
    cam_id = request.args.get('camera_id', type=int) or request.form.get('camera_id', type=int) or 1
    return redirect(url_for('api.api_simulate', kind='violation', camera_id=cam_id), code=307)
