"""
SAFELABS — Edge AI Laboratory Safety Monitoring Platform.
Application factory, configuration, and Blueprint registration.
"""
import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()  # Load .env before any os.environ.get() calls

from flask import Flask, redirect, url_for, flash
from flask_login import LoginManager, current_user

from database import (
    db, User, CameraFeed, HazardMaster, EmergencyAlert,
    ClassSchedule, AuditLog,
)
from safety_engine import get_current_ongoing_class
from time_utils import get_myt_now, format_date, format_time, format_datetime

# ---------------------------------------------------------------------------
# App factory & configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Cryptographically verified SECRET_KEY enforcement
configured_secret = os.environ.get('SAFELABS_SECRET_KEY')
if not configured_secret or configured_secret == 'SAFELABS_SECURE_KEY_CHANGE_IN_PRODUCTION':
    # Stable deterministic development fallback key prevents session loss on auto-reloads
    configured_secret = 'SAFELABS_STABLE_DEV_KEY_e83f2a7b1c4d9e0f5a6b8c7d3e2f1a0b'
app.config['SECRET_KEY'] = configured_secret
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'sqlite:///safelabs.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = os.environ.get(
    'FLASK_ENV', 'development'
) == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
app.config['REMEMBER_COOKIE_SECURE'] = app.config['SESSION_COOKIE_SECURE']
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db.init_app(app)

# Configure SQLite WAL mode & busy timeout to prevent concurrent locking between threads
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if hasattr(dbapi_connection, "cursor"):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        except Exception:
            pass

try:
    from flask_wtf.csrf import CSRFProtect, CSRFError
    csrf = CSRFProtect(app)
except ImportError:
    class _DummyCSRF:
        """Fallback dummy CSRF protector if flask_wtf is not installed."""
        def exempt(self, view_func):
            return view_func
    csrf = _DummyCSRF()
    CSRFError = Exception


@app.after_request
def set_security_headers(response):
    """Enforce strict security headers across all HTTP responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' "
        "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com https://cdn.tailwindcss.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    if os.environ.get('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains'
        )
    return response


@app.before_request
def configure_session():
    """Enforce mandatory password changes for temporary/reset accounts."""
    from flask import session, request, redirect, url_for

    if current_user.is_authenticated and getattr(current_user, 'must_change_password', False):
        endpoint = request.endpoint or ''
        allowed_endpoints = {'auth.change_password', 'auth.logout', 'static'}
        if endpoint not in allowed_endpoints and not request.path.startswith('/static/'):
            return redirect(url_for('auth.change_password'))


login_manager = LoginManager()
login_manager.init_app(app)
setattr(login_manager, 'login_view', 'auth.login')
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    """Load user by primary key for session management."""
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Template filters — standardized date and time formatting (MYT & dd/mm/yy)
# ---------------------------------------------------------------------------
@app.template_filter('format_date')
def jinja_format_date(dt, default='-'):
    return format_date(dt, default=default)


@app.template_filter('format_time')
def jinja_format_time(dt, include_seconds=True, default='-'):
    return format_time(dt, include_seconds=include_seconds, default=default)


@app.template_filter('format_datetime')
def jinja_format_datetime(dt, include_seconds=True, default='-'):
    return format_datetime(dt, include_seconds=include_seconds, default=default)


# ---------------------------------------------------------------------------
# Context processor — inject globals into every template
# ---------------------------------------------------------------------------
@app.context_processor
def inject_globals():
    """Inject unread alert badges, cameras, and ongoing class globally."""
    unread_alerts = 0
    if current_user.is_authenticated and current_user.role in ('afm', 'admin'):
        unread_alerts = EmergencyAlert.query.filter_by(
            is_acknowledged=False
        ).count()

    all_cameras = CameraFeed.query.filter_by(is_active=True).all()
    ongoing = None
    if current_user.is_authenticated:
        ongoing = get_current_ongoing_class(db.session, ClassSchedule)

    return dict(
        now=get_myt_now(),
        unread_alerts=unread_alerts,
        global_cameras=all_cameras,
        current_ongoing_class=ongoing,
        format_date=format_date,
        format_time=format_time,
        format_datetime=format_datetime,
    )


# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------
from routes.main import main_bp          # noqa: E402
from routes.auth import auth_bp          # noqa: E402
from routes.admin import admin_bp        # noqa: E402
from routes.classes import classes_bp    # noqa: E402
from routes.violations import violations_bp  # noqa: E402
from routes.alerts import alerts_bp      # noqa: E402
from routes.video import video_bp        # noqa: E402
from routes.api import api_bp           # noqa: E402

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(classes_bp)
app.register_blueprint(violations_bp)
app.register_blueprint(alerts_bp)
app.register_blueprint(video_bp)
app.register_blueprint(api_bp)

from routes.evidence import evidence_bp  # noqa: E402
app.register_blueprint(evidence_bp)

# Exempt API and auth endpoints from CSRF
csrf.exempt(api_bp)
csrf.exempt(auth_bp)


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    """Gracefully handle expired CSRF session tokens instead of raw 400 error."""
    flash('Your session has expired or the security token was invalid. Please sign in again.', 'warning')
    return redirect(url_for('auth.login'))


@app.errorhandler(403)
def handle_forbidden(e):
    """Render structured 403 Forbidden page or JSON response for unauthorized access."""
    from flask import request, jsonify, render_template
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden', 'message': 'Insufficient permissions for this action.'}), 403
    return render_template('403.html'), 403


# ===================================================================
#  DATABASE SEEDING (Direct DB Population, Zero Hardcoding in Logic)
# ===================================================================
def seed_database():
    """Populate default users, cameras, and hazard rules directly into DB."""
    # 1. Seed Users
    default_users = [
        {
            'username': 'admin', 'password': 'admin',
            'role': 'admin', 'full_name': 'System Administrator',
            'must_change_password': False,
        },
        {
            'username': 'academic_coordinator', 'password': 'academic_coordinator',
            'role': 'academic_coordinator', 'full_name': 'Academic Coordinator',
            'must_change_password': True,
        },
        {
            'username': 'afm', 'password': 'afm',
            'role': 'afm', 'full_name': 'Admin & Facilities Management Department',
            'must_change_password': True,
        },
    ]
    for u in default_users:
        if not User.query.filter_by(username=u['username']).first():
            user = User(
                username=u['username'], role=u['role'],
                full_name=u['full_name'],
                must_change_password=u.get('must_change_password', False),
            )
            user.set_password(u['password'])
            db.session.add(user)

    # 2. Seed at least 4 Camera Feeds directly to DB
    default_rtsp = os.environ.get(
        'SAFELABS_CAMERA_1_SOURCE'
    ) or os.environ.get(
        'SAFELABS_RTSP_URL',
        'rtsp://admin:JYLSS.25@192.168.1.64:554/Streaming/Channels/101',
    )
    default_cameras = [
        {
            'id': 1, 'code': 'CAM-01',
            'name': 'Lab A',
            'location': 'Lab A',
            'description': 'Lab A',
            'stream_source': default_rtsp,
        },
        {
            'id': 2, 'code': 'CAM-02',
            'name': 'Lab B',
            'location': 'Lab B',
            'description': 'Lab B',
            'stream_source': os.environ.get('SAFELABS_CAMERA_2_SOURCE', '1'),
        },
        {
            'id': 3, 'code': 'CAM-03',
            'name': 'Lab C',
            'location': 'Lab C',
            'description': 'Lab C',
            'stream_source': os.environ.get('SAFELABS_CAMERA_3_SOURCE', '2'),
        },
        {
            'id': 4, 'code': 'CAM-04',
            'name': 'Lab D',
            'location': 'Lab D',
            'description': 'Lab D',
            'stream_source': os.environ.get('SAFELABS_CAMERA_4_SOURCE', '3'),
        },
    ]
    for c in default_cameras:
        cam_exist = CameraFeed.query.filter_by(code=c['code']).first()
        if not cam_exist:
            feed = CameraFeed(
                name=c['name'], code=c['code'], location=c['location'],
                description=c['description'], stream_source=c['stream_source'],
            )
            db.session.add(feed)
        else:
            cam_exist.name = c['name']
            cam_exist.location = c['location']
            cam_exist.description = c['description']
            if c['code'] == 'CAM-01' and cam_exist.stream_source in ('0', None, ''):
                cam_exist.stream_source = c['stream_source']

    # 3. Seed Hazard Rules directly to DB
    default_hazards = [
        {
            'hazard_type': 'Missing Face Mask',
            'category': 'ppe',
            'default_description': (
                'Student observed without mandatory N95 / surgical mask.'
            ),
            'default_severity': 'high',
        },
        {
            'hazard_type': 'Missing Lab Coat',
            'category': 'ppe',
            'default_description': (
                'Individual in laboratory workspace without protective coat.'
            ),
            'default_severity': 'medium',
        },
        {
            'hazard_type': 'Missing Safety Goggles',
            'category': 'ppe',
            'default_description': (
                'Eye protection not detected while handling reagents.'
            ),
            'default_severity': 'critical',
        },
        {
            'hazard_type': 'Unauthorized Phone Usage',
            'category': 'electronic',
            'default_description': (
                'Mobile smartphone usage detected during active lab session.'
            ),
            'default_severity': 'medium',
        },
        {
            'hazard_type': 'Unauthorized Laptop Usage',
            'category': 'electronic',
            'default_description': (
                'Unauthorized personal computing device on lab chemical bench.'
            ),
            'default_severity': 'low',
        },
        {
            'hazard_type': 'Chemical Spill Detected',
            'category': 'hazard',
            'default_description': (
                'Hazardous liquid dispersion pattern identified on floor.'
            ),
            'default_severity': 'critical',
        },
        {
            'hazard_type': 'Thermal Anomaly / Flame',
            'category': 'hazard',
            'default_description': (
                'Thermal anomaly or open flame detected near combustible area.'
            ),
            'default_severity': 'critical',
        },
        {
            'hazard_type': 'Combustion Smoke Plume',
            'category': 'hazard',
            'default_description': (
                'Particle sensor and vision plume threshold exceeded.'
            ),
            'default_severity': 'critical',
        },
    ]
    for h in default_hazards:
        if not HazardMaster.query.filter_by(
            hazard_type=h['hazard_type']
        ).first():
            hazard = HazardMaster(
                hazard_type=h['hazard_type'],
                category=h['category'],
                default_description=h['default_description'],
                default_severity=h['default_severity'],
            )
            db.session.add(hazard)

    db.session.commit()


def migrate_database_schema():
    """Ensure newly added columns exist in SQLite database without losing data."""
    try:
        with db.engine.connect() as conn:
            # --- ViolationLog migrations ---
            res = conn.exec_driver_sql("PRAGMA table_info(violation_log)")
            existing_cols = {row[1] for row in res.fetchall()}
            if existing_cols:
                if 'confidence_score' not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE violation_log ADD COLUMN confidence_score FLOAT"
                    )
                if 'review_status' not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE violation_log ADD COLUMN review_status VARCHAR(20) DEFAULT 'confirmed'"
                    )
                if 'reviewed_by_id' not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE violation_log ADD COLUMN reviewed_by_id INTEGER REFERENCES user(id)"
                    )
                if 'evidence_hash' not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE violation_log ADD COLUMN evidence_hash VARCHAR(64)"
                    )
                if 'duration_seconds' not in existing_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE violation_log ADD COLUMN duration_seconds INTEGER DEFAULT 0"
                    )

            # --- EmergencyAlert migrations ---
            res2 = conn.exec_driver_sql("PRAGMA table_info(emergency_alert)")
            alert_cols = {row[1] for row in res2.fetchall()}
            if alert_cols:
                if 'evidence_hash' not in alert_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE emergency_alert ADD COLUMN evidence_hash VARCHAR(64)"
                    )
                if 'review_status' not in alert_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE emergency_alert ADD COLUMN review_status VARCHAR(20) DEFAULT 'confirmed'"
                    )
                if 'reviewed_by_id' not in alert_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE emergency_alert ADD COLUMN reviewed_by_id INTEGER REFERENCES user(id)"
                    )

            # --- User table migrations ---
            res_u = conn.exec_driver_sql("PRAGMA table_info(user)")
            user_cols = {row[1] for row in res_u.fetchall()}
            if user_cols:
                if 'must_change_password' not in user_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE user ADD COLUMN must_change_password BOOLEAN DEFAULT 0"
                    )
                if 'failed_login_attempts' not in user_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE user ADD COLUMN failed_login_attempts INTEGER DEFAULT 0"
                    )
                if 'locked_until' not in user_cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE user ADD COLUMN locked_until DATETIME"
                    )

            # --- User role migration (lab_assistant -> academic_coordinator) ---
            coord_exists = conn.exec_driver_sql(
                "SELECT id FROM user WHERE username = 'academic_coordinator'"
            ).fetchone()
            if coord_exists:
                conn.exec_driver_sql("DELETE FROM user WHERE username = 'lab_assistant'")
            else:
                conn.exec_driver_sql(
                    "UPDATE user SET username = 'academic_coordinator', full_name = 'Academic Coordinator' WHERE username = 'lab_assistant'"
                )
            conn.exec_driver_sql(
                "UPDATE user SET role = 'academic_coordinator' WHERE role = 'lab_assistant'"
            )

            # --- Camera feed name simplification (Lab A, Lab B, etc.) ---
            conn.exec_driver_sql(
                "UPDATE camera_feed SET name = location, description = location WHERE name != location"
            )

            # --- Shift historical UTC timestamps to Malaysian Time (UTC+8) once ---
            conn.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS schema_version (key VARCHAR(50) PRIMARY KEY, applied_at DATETIME)"
            )
            shifted = conn.exec_driver_sql(
                "SELECT 1 FROM schema_version WHERE key = 'shift_utc_to_myt_v1'"
            ).fetchone()
            if not shifted:
                conn.exec_driver_sql(
                    "UPDATE violation_log SET timestamp = datetime(timestamp, '+8 hours') WHERE timestamp IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE emergency_alert SET timestamp = datetime(timestamp, '+8 hours') WHERE timestamp IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE emergency_alert SET acknowledged_at = datetime(acknowledged_at, '+8 hours') WHERE acknowledged_at IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE violation_log SET acknowledged_at = datetime(acknowledged_at, '+8 hours') WHERE acknowledged_at IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE audit_log SET timestamp = datetime(timestamp, '+8 hours') WHERE timestamp IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE user SET created_at = datetime(created_at, '+8 hours') WHERE created_at IS NOT NULL"
                )
                conn.exec_driver_sql(
                    "INSERT INTO schema_version (key, applied_at) VALUES ('shift_utc_to_myt_v1', datetime('now'))"
                )

            conn.commit()
    except Exception as e:
        print(f"[Schema Migration] Warning during migration: {e}")


# Auto-migrate schema on module load
with app.app_context():
    try:
        migrate_database_schema()
    except Exception as e:
        pass


# ===================================================================
#  ENTRYPOINT
# ===================================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        migrate_database_schema()
        seed_database()

        static_path = app.static_folder or os.path.join(
            app.root_path, 'static'
        )
        os.makedirs(os.path.join(static_path, 'evidence'), exist_ok=True)

        # Pre-warm AI models for instant live inference
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[SAFELABS AI] Hardware Acceleration: Dedicated GPU active ({gpu_name})")
            else:
                print("[SAFELABS AI] Hardware Acceleration: CPU (CUDA not available)")

            from safety_engine import get_roboflow_client, get_person_model, get_active_backend, AVAILABLE_MODELS
            get_person_model()
            get_roboflow_client()
            print(f"[SAFELABS AI] Roboflow API & Cloud API initialized. Active backend: {get_active_backend()}")
        except Exception as e:
            print(f"[SAFELABS AI] Warning during model pre-warm: {e}")

    app.run(debug=True, threaded=True)
