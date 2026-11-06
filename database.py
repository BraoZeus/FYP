"""
Database models for SAFELABS safety monitoring platform.
Configured for SQLite/PostgreSQL with Flask-SQLAlchemy.
"""
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from time_utils import get_myt_now

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model with role-based access control."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    email = db.Column(db.String(150), nullable=True)
    is_active_account = db.Column(db.Boolean, default=True)
    must_change_password = db.Column(db.Boolean, default=False)
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())

    def __init__(self, username=None, password_hash=None, role=None,
                 full_name=None, email=None, is_active_account=True,
                 must_change_password=False,
                 failed_login_attempts=0, locked_until=None,
                 **kwargs):
        super().__init__(**kwargs)
        if username is not None:
            self.username = username
        if password_hash is not None:
            self.password_hash = password_hash
        if role is not None:
            self.role = role
        if full_name is not None:
            self.full_name = full_name
        if email is not None:
            self.email = email
        self.is_active_account = is_active_account
        self.must_change_password = must_change_password
        self.failed_login_attempts = failed_login_attempts
        self.locked_until = locked_until
        for key, value in kwargs.items():
            setattr(self, key, value)

    def is_locked(self):
        """Check if account is temporarily locked due to consecutive failed attempts."""
        if self.locked_until and self.locked_until > get_myt_now():
            return True
        return False

    def lock_account(self, minutes=15):
        """Temporarily lock the account for specified minutes."""
        self.locked_until = get_myt_now() + timedelta(minutes=minutes)

    def unlock_account(self):
        """Reset failed attempts and clear lock."""
        self.failed_login_attempts = 0
        self.locked_until = None

    def set_password(self, password):
        self.password_hash = generate_password_hash(
            password, method='pbkdf2:sha256', salt_length=16
        )

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def role_display(self):
        return {
            'admin': 'Administrator',
            'academic_coordinator': 'Academic Coordinator',
            'afm': 'Admin & Facilities Management Department',
        }.get(self.role, self.role)


class CameraFeed(db.Model):
    """Camera hardware/stream source configuration stored in database."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True, nullable=False)
    location = db.Column(db.String(150), nullable=False)
    stream_source = db.Column(db.String(255), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(20), default='online')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())

    def __init__(self, name=None, code=None, location=None,
                 stream_source=None, description=None, status='online',
                 is_active=True, **kwargs):
        super().__init__(**kwargs)
        if name is not None:
            self.name = name
        if code is not None:
            self.code = code
        if location is not None:
            self.location = location
        if stream_source is not None:
            self.stream_source = stream_source
        if description is not None:
            self.description = description
        self.status = status
        self.is_active = is_active
        for key, value in kwargs.items():
            setattr(self, key, value)


class HazardMaster(db.Model):
    """Configurable hazard rules and default descriptions in database."""
    id = db.Column(db.Integer, primary_key=True)
    hazard_type = db.Column(db.String(100), unique=True, nullable=False)
    category = db.Column(db.String(50), default='ppe')
    default_description = db.Column(db.String(500), nullable=True)
    default_severity = db.Column(db.String(20), default='medium')
    is_active = db.Column(db.Boolean, default=True)

    def __init__(self, hazard_type=None, category='ppe',
                 default_description=None, default_severity='medium',
                 is_active=True, **kwargs):
        super().__init__(**kwargs)
        if hazard_type is not None:
            self.hazard_type = hazard_type
        self.category = category
        if default_description is not None:
            self.default_description = default_description
        self.default_severity = default_severity
        self.is_active = is_active
        for key, value in kwargs.items():
            setattr(self, key, value)


class ClassSchedule(db.Model):
    """Laboratory class schedule and assignment."""
    id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(150), nullable=False)
    class_code = db.Column(db.String(30), nullable=False)
    lab_room = db.Column(db.String(50), nullable=False)
    instructor = db.Column(db.String(150), nullable=False)
    day_of_week = db.Column(db.String(20), nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    semester = db.Column(db.String(50), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=False
    )
    created_at = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())
    updated_at = db.Column(
        db.DateTime, default=get_myt_now, onupdate=get_myt_now, server_default=db.func.now()
    )

    created_by = db.relationship('User', foreign_keys=[created_by_id])
    monitoring = db.relationship(
        'MonitoringConfig', backref='class_schedule',
        uselist=False, cascade='all, delete-orphan'
    )

    def __init__(self, class_name=None, class_code=None, lab_room=None,
                 instructor=None, day_of_week=None, start_time=None,
                 end_time=None, start_date=None, end_date=None,
                 semester=None, is_active=True,
                 created_by_id=None, **kwargs):
        super().__init__(**kwargs)
        if class_name is not None:
            self.class_name = class_name
        if class_code is not None:
            self.class_code = class_code
        if lab_room is not None:
            self.lab_room = lab_room
        if instructor is not None:
            self.instructor = instructor
        if day_of_week is not None:
            self.day_of_week = day_of_week
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
        if semester is not None:
            self.semester = semester
        self.is_active = is_active
        if created_by_id is not None:
            self.created_by_id = created_by_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def is_ongoing(self):
        now = get_myt_now()
        current_day = now.strftime('%A')
        current_time = now.time()
        current_date = now.date()

        in_date_range = True
        if self.start_date and current_date < self.start_date:
            in_date_range = False
        if self.end_date and current_date > self.end_date:
            in_date_range = False

        return (
            self.is_active
            and in_date_range
            and self.day_of_week == current_day
            and self.start_time <= current_time <= self.end_time
        )


class MonitoringConfig(db.Model):
    """Configurable edge AI safety checks per class session."""
    id = db.Column(db.Integer, primary_key=True)
    class_schedule_id = db.Column(
        db.Integer, db.ForeignKey('class_schedule.id'),
        nullable=False, unique=True
    )
    monitor_full_ppe = db.Column(db.Boolean, default=True)
    monitor_phones = db.Column(db.Boolean, default=True)
    monitor_laptops = db.Column(db.Boolean, default=True)
    monitor_face_mask = db.Column(db.Boolean, default=True)
    monitor_lab_coat = db.Column(db.Boolean, default=True)
    monitor_goggles = db.Column(db.Boolean, default=False)
    updated_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=False
    )
    updated_at = db.Column(
        db.DateTime, default=get_myt_now, onupdate=get_myt_now, server_default=db.func.now()
    )

    updated_by = db.relationship('User', foreign_keys=[updated_by_id])

    def __init__(self, class_schedule_id=None, monitor_full_ppe=True,
                 monitor_phones=True, monitor_laptops=True,
                 monitor_face_mask=True, monitor_lab_coat=True,
                 monitor_goggles=False, updated_by_id=None, **kwargs):
        super().__init__(**kwargs)
        if class_schedule_id is not None:
            self.class_schedule_id = class_schedule_id
        self.monitor_full_ppe = monitor_full_ppe
        self.monitor_phones = monitor_phones
        self.monitor_laptops = monitor_laptops
        self.monitor_face_mask = monitor_face_mask
        self.monitor_lab_coat = monitor_lab_coat
        self.monitor_goggles = monitor_goggles
        if updated_by_id is not None:
            self.updated_by_id = updated_by_id
        for key, value in kwargs.items():
            setattr(self, key, value)


class ViolationLog(db.Model):
    """Laboratory safety detection event logs."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())
    hazard_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(500), nullable=True)
    severity = db.Column(db.String(20), default='medium')
    confidence_score = db.Column(db.Float, nullable=True)
    duration_seconds = db.Column(db.Integer, default=0, nullable=True)
    image_path = db.Column(db.String(255), nullable=True)
    evidence_hash = db.Column(db.String(64), nullable=True)
    video_clip_path = db.Column(db.String(255), nullable=True)
    camera_id = db.Column(
        db.Integer, db.ForeignKey('camera_feed.id'), nullable=True
    )
    class_schedule_id = db.Column(
        db.Integer, db.ForeignKey('class_schedule.id'), nullable=True
    )
    is_acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolution_note = db.Column(db.String(500), nullable=True)
    review_status = db.Column(
        db.String(20), default='pending', nullable=False
    )
    reviewed_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )

    camera = db.relationship('CameraFeed', foreign_keys=[camera_id])
    class_schedule = db.relationship(
        'ClassSchedule', foreign_keys=[class_schedule_id]
    )
    acknowledged_by = db.relationship(
        'User', foreign_keys=[acknowledged_by_id]
    )
    reviewed_by = db.relationship(
        'User', foreign_keys=[reviewed_by_id]
    )

    def __init__(self, hazard_type=None, description=None,
                 severity='medium', confidence_score=None,
                 duration_seconds=0,
                 image_path=None, evidence_hash=None,
                 video_clip_path=None,
                 camera_id=None, class_schedule_id=None,
                 is_acknowledged=False, acknowledged_by_id=None,
                 acknowledged_at=None, resolution_note=None,
                 review_status='pending', reviewed_by_id=None,
                 **kwargs):
        super().__init__(**kwargs)
        if hazard_type is not None:
            self.hazard_type = hazard_type
        if description is not None:
            self.description = description
        self.severity = severity
        self.confidence_score = confidence_score
        self.duration_seconds = duration_seconds
        if image_path is not None:
            self.image_path = image_path
        self.evidence_hash = evidence_hash
        if video_clip_path is not None:
            self.video_clip_path = video_clip_path
        if camera_id is not None:
            self.camera_id = camera_id
        if class_schedule_id is not None:
            self.class_schedule_id = class_schedule_id
        self.is_acknowledged = is_acknowledged
        if acknowledged_by_id is not None:
            self.acknowledged_by_id = acknowledged_by_id
        if acknowledged_at is not None:
            self.acknowledged_at = acknowledged_at
        if resolution_note is not None:
            self.resolution_note = resolution_note
        self.review_status = review_status
        if reviewed_by_id is not None:
            self.reviewed_by_id = reviewed_by_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def confidence_pct(self):
        """Return confidence as a percentage string (e.g. '94.3%')."""
        if self.confidence_score is not None:
            return f"{self.confidence_score * 100:.1f}%"
        return None

    @property
    def is_false_positive(self):
        return self.review_status == 'false_positive'

    @property
    def severity_color(self):
        return {
            'low': 'sky',
            'medium': 'amber',
            'high': 'orange',
            'critical': 'red',
        }.get(self.severity, 'gray')


class EmergencyAlert(db.Model):
    """Critical safety alerts (e.g., fire, smoke, hazardous spills)."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    evidence_hash = db.Column(db.String(64), nullable=True)
    camera_id = db.Column(
        db.Integer, db.ForeignKey('camera_feed.id'), nullable=True
    )
    is_acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolution_note = db.Column(db.String(500), nullable=True)
    review_status = db.Column(
        db.String(20), default='pending', nullable=False
    )
    reviewed_by_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )

    camera = db.relationship('CameraFeed', foreign_keys=[camera_id])
    acknowledged_by = db.relationship(
        'User', foreign_keys=[acknowledged_by_id]
    )
    reviewed_by = db.relationship(
        'User', foreign_keys=[reviewed_by_id]
    )

    def __init__(self, alert_type=None, message=None, image_path=None,
                 evidence_hash=None, camera_id=None,
                 is_acknowledged=False,
                 acknowledged_by_id=None, acknowledged_at=None,
                 resolution_note=None,
                 review_status='pending', reviewed_by_id=None, **kwargs):
        super().__init__(**kwargs)
        if alert_type is not None:
            self.alert_type = alert_type
        if message is not None:
            self.message = message
        if image_path is not None:
            self.image_path = image_path
        self.evidence_hash = evidence_hash
        if camera_id is not None:
            self.camera_id = camera_id
        self.is_acknowledged = is_acknowledged
        if acknowledged_by_id is not None:
            self.acknowledged_by_id = acknowledged_by_id
        if acknowledged_at is not None:
            self.acknowledged_at = acknowledged_at
        if resolution_note is not None:
            self.resolution_note = resolution_note
        self.review_status = review_status
        if reviewed_by_id is not None:
            self.reviewed_by_id = reviewed_by_id
        for key, value in kwargs.items():
            setattr(self, key, value)


class AuditLog(db.Model):
    """Security audit trail for login, logout, and evidence access events."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=get_myt_now, server_default=db.func.now())
    user_id = db.Column(
        db.Integer, db.ForeignKey('user.id'), nullable=True
    )
    username = db.Column(db.String(100), nullable=True)
    action = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    resource_type = db.Column(db.String(50), nullable=True)
    resource_id = db.Column(db.Integer, nullable=True)
    details = db.Column(db.String(500), nullable=True)
    success = db.Column(db.Boolean, default=True)
    event_type = db.Column(db.String(20), default='INFO', nullable=True)

    user = db.relationship('User', foreign_keys=[user_id])

    def __init__(self, action=None, user_id=None, username=None,
                 ip_address=None, user_agent=None, resource_type=None,
                 resource_id=None, details=None, success=True, event_type=None, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.user_id = user_id
        self.username = username
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details = details
        self.success = success
        if event_type is None:
            if not success or (action and any(k in action.lower() for k in ['fail', 'lock', 'tamper', 'error'])):
                event_type = 'ERROR'
            elif (action and any(k in action.lower() for k in ['warn', 'threshold', 'confidence', 'exceed'])) or \
                 (details and any(k in details.lower() for k in ['threshold', 'confidence', 'warning', 'offline', 'adjust'])):
                event_type = 'WARNING'
            else:
                event_type = 'INFO'
        self.event_type = event_type

    @property
    def computed_event_type(self):
        return self.event_type or ('ERROR' if not self.success else 'INFO')
