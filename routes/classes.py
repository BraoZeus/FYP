"""
Class schedule, monitoring config, and ongoing classes Blueprint.
"""
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from database import db, ClassSchedule, MonitoringConfig, CameraFeed
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR, VALID_ROLES
from audit_utils import log_audit_event
from safety_engine import get_objective2_model_catalog, get_hardware_telemetry
from time_utils import get_myt_now

classes_bp = Blueprint('classes', __name__)

DAYS_OF_WEEK = [
    'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday',
]


# ───────────────────────── Class Schedule CRUD ─────────────────────────

@classes_bp.route('/class_schedule')
@classes_bp.route('/classes')
@role_required(ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR)
def classes_list():
    """Consolidated Class Schedule view: [Active Session] and [Timetable]."""
    tab = request.args.get('tab', 'active')
    semester = request.args.get('semester', '')

    now = get_myt_now()
    current_day = now.strftime('%A')
    current_time = now.time()
    current_date = now.date()

    ongoing = ClassSchedule.query.filter_by(
        is_active=True
    ).filter(
        ClassSchedule.day_of_week == current_day,
        ClassSchedule.start_time <= current_time,
        ClassSchedule.end_time >= current_time,
    ).all()
    ongoing = [
        c for c in ongoing
        if (not c.start_date or current_date >= c.start_date) and (not c.end_date or current_date <= c.end_date)
    ]

    upcoming = ClassSchedule.query.filter_by(
        is_active=True
    ).filter(
        ClassSchedule.day_of_week == current_day,
        ClassSchedule.start_time > current_time,
    ).order_by(ClassSchedule.start_time).all()
    upcoming = [
        c for c in upcoming
        if (not c.start_date or current_date >= c.start_date) and (not c.end_date or current_date <= c.end_date)
    ]

    q = ClassSchedule.query
    if semester:
        q = q.filter_by(semester=semester)
    classes = q.order_by(
        ClassSchedule.day_of_week, ClassSchedule.start_time
    ).all()

    semesters = [
        s[0] for s in
        db.session.query(ClassSchedule.semester).distinct().all()
    ]

    cameras = CameraFeed.query.filter_by(is_active=True).all()

    return render_template(
        'classes.html',
        active_tab=tab,
        ongoing=ongoing,
        upcoming=upcoming,
        classes=classes,
        semesters=semesters,
        current_semester=semester,
        now=now,
        cameras=cameras,
    )


@classes_bp.route('/class_schedule/create', methods=['GET', 'POST'])
@classes_bp.route('/classes/create', methods=['GET', 'POST'])
@role_required(ROLE_ADMIN, ROLE_ACADEMIC_COORDINATOR)
def class_create():
    """Create a new class schedule (Admin and Academic Coordinator only)."""
    if request.method == 'POST':
        return _handle_class_form(mode='create')

    return render_template(
        'class_form.html', mode='create',
        form_data={}, days=DAYS_OF_WEEK,
    )


@classes_bp.route('/class_schedule/<int:class_id>/edit', methods=['GET', 'POST'])
@classes_bp.route('/classes/<int:class_id>/edit', methods=['GET', 'POST'])
@role_required(ROLE_ADMIN, ROLE_ACADEMIC_COORDINATOR)
def class_edit(class_id):
    """Edit class schedule and embedded safety monitoring rules (Admin and Academic Coordinator only)."""
    schedule = ClassSchedule.query.get_or_404(class_id)

    if request.method == 'POST':
        return _handle_class_form(mode='edit', schedule=schedule)

    return render_template(
        'class_form.html', mode='edit',
        schedule=schedule, form_data={}, days=DAYS_OF_WEEK,
    )


@classes_bp.route('/class_schedule/<int:class_id>/delete', methods=['POST'])
@classes_bp.route('/classes/<int:class_id>/delete', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_ACADEMIC_COORDINATOR)
def class_delete(class_id):
    """Delete a class schedule (Admin and Academic Coordinator only)."""
    schedule = ClassSchedule.query.get_or_404(class_id)
    name = schedule.class_name
    db.session.delete(schedule)
    db.session.commit()

    log_audit_event(
        'schedule_delete',
        resource_type='class_schedule',
        resource_id=class_id,
        details=f"Deleted class '{name}'"
    )

    flash(f'Class "{name}" deleted.', 'success')
    return redirect(url_for('classes.classes_list'))


def _handle_class_form(mode, schedule=None):
    """Shared create/edit handler for class schedules and monitoring."""
    f = request.form
    class_name = f.get('class_name', '').strip()
    class_code = f.get('class_code', '').strip()
    lab_room = f.get('lab_room', '').strip()
    instructor = f.get('instructor', '').strip()
    day_of_week = f.get('day_of_week', '')
    start_str = f.get('start_time', '')
    end_str = f.get('end_time', '')
    start_date_str = f.get('start_date', '').strip()
    end_date_str = f.get('end_date', '').strip()
    semester = f.get('semester', '').strip()

    errors = []
    if not class_name:
        errors.append('Class name is required.')
    if not class_code:
        errors.append('Class code is required.')
    if not lab_room:
        errors.append('Lab room is required.')
    if not instructor:
        errors.append('Instructor is required.')
    if day_of_week not in DAYS_OF_WEEK:
        errors.append('Select a valid day.')
    if not semester:
        errors.append('Semester is required.')

    start_time = end_time = None
    try:
        start_time = datetime.strptime(start_str, '%H:%M').time()
        end_time = datetime.strptime(end_str, '%H:%M').time()
        if start_time >= end_time:
            errors.append('End time must be after start time.')
    except ValueError:
        errors.append('Invalid time format.')

    start_date = end_date = None
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        except ValueError:
            errors.append('Invalid start date format (use YYYY-MM-DD).')
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            errors.append('Invalid end date format (use YYYY-MM-DD).')
    if start_date and end_date and start_date > end_date:
        errors.append('Semester end date cannot be earlier than start date.')

    if errors:
        for e in errors:
            flash(e, 'error')
        return render_template(
            'class_form.html', mode=mode,
            schedule=schedule, form_data=f, days=DAYS_OF_WEEK,
        )

    # Intelligent Date vs Day mismatch verification
    if start_date and day_of_week in DAYS_OF_WEEK:
        start_day_name = start_date.strftime('%A')
        if start_day_name != day_of_week and f.get('confirm_day_mismatch') != '1':
            day_mismatch_warning = {
                'start_date_str': start_date_str,
                'start_day_name': start_day_name,
                'selected_day': day_of_week,
            }
            flash(
                f"Schedule Verification: Start date ({start_date_str}) is a {start_day_name}, but recurring class is set to {day_of_week}. "
                f"Please confirm whether this is the semester opening date or if the class day should be changed.",
                'warning'
            )
            return render_template(
                'class_form.html', mode=mode,
                schedule=schedule, form_data=f, days=DAYS_OF_WEEK,
                day_mismatch_warning=day_mismatch_warning,
            )

    # Safety triggers: full PPE (mask, coat, gloves combined), laptops, phones. Goggles removed.
    is_full_ppe = f.get('monitor_full_ppe') == 'on'
    monitor_phones = f.get('monitor_phones') == 'on'
    monitor_laptops = f.get('monitor_laptops') == 'on'

    if mode == 'create':
        schedule = ClassSchedule(
            class_name=class_name, class_code=class_code,
            lab_room=lab_room, instructor=instructor,
            day_of_week=day_of_week, start_time=start_time,
            end_time=end_time, start_date=start_date, end_date=end_date,
            semester=semester, created_by_id=current_user.id,
        )
        db.session.add(schedule)
        db.session.flush()
        m_cfg = MonitoringConfig(
            class_schedule_id=schedule.id,
            monitor_full_ppe=is_full_ppe,
            monitor_phones=monitor_phones,
            monitor_laptops=monitor_laptops,
            monitor_face_mask=is_full_ppe,
            monitor_lab_coat=is_full_ppe,
            monitor_goggles=False,
            updated_by_id=current_user.id,
        )
        db.session.add(m_cfg)
    else:
        assert schedule is not None
        schedule.class_name = class_name
        schedule.class_code = class_code
        schedule.lab_room = lab_room
        schedule.instructor = instructor
        schedule.day_of_week = day_of_week
        schedule.start_time = start_time
        schedule.end_time = end_time
        schedule.start_date = start_date
        schedule.end_date = end_date
        schedule.semester = semester
        schedule.is_active = f.get('is_active') == 'on'

        m_rel = schedule.monitoring
        if not m_rel:
            m_rel = MonitoringConfig(
                class_schedule_id=schedule.id,
                updated_by_id=current_user.id,
            )
            schedule.monitoring = m_rel
        m_rel.monitor_full_ppe = is_full_ppe
        m_rel.monitor_face_mask = is_full_ppe
        m_rel.monitor_lab_coat = is_full_ppe
        m_rel.monitor_goggles = False
        m_rel.monitor_phones = monitor_phones
        m_rel.monitor_laptops = monitor_laptops
        m_rel.updated_by_id = current_user.id

    db.session.commit()

    log_audit_event(
        'schedule_create' if mode == 'create' else 'schedule_edit',
        resource_type='class_schedule',
        resource_id=schedule.id,
        details=f"{'Created' if mode == 'create' else 'Updated'} class '{class_name}' ({class_code}) in {lab_room}"
    )

    act = "created" if mode == "create" else "updated"
    flash(f'Class "{class_name}" {act}.', 'success')
    return redirect(url_for('classes.classes_list'))


# ───────────────────────── Monitoring Config ─────────────────────────

@classes_bp.route('/detection_rules')
@classes_bp.route('/monitoring')
@role_required(ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR)
def monitoring_list():
    """Class monitoring rules overview (Admin, AFM, and Academic Coordinator)."""
    classes = ClassSchedule.query.filter_by(is_active=True).order_by(
        ClassSchedule.day_of_week, ClassSchedule.start_time,
    ).all()
    return render_template(
        'monitoring.html',
        classes=classes,
        model_catalog=get_objective2_model_catalog(),
        hw_status=get_hardware_telemetry(),
    )


@classes_bp.route('/detection_rules/<int:class_id>', methods=['POST'])
@classes_bp.route('/monitoring/<int:class_id>', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def monitoring_update(class_id):
    """Update active safety hazard triggers for a class (Admin and AFM only)."""
    schedule = ClassSchedule.query.get_or_404(class_id)
    config = schedule.monitoring

    if not config:
        config = MonitoringConfig(
            class_schedule_id=class_id,
            updated_by_id=current_user.id,
        )
        db.session.add(config)

    is_full_ppe = request.form.get('monitor_full_ppe') == 'on'
    config.monitor_full_ppe = is_full_ppe
    config.monitor_face_mask = is_full_ppe
    config.monitor_lab_coat = is_full_ppe
    config.monitor_goggles = False
    config.monitor_phones = request.form.get('monitor_phones') == 'on'
    config.monitor_laptops = request.form.get('monitor_laptops') == 'on'
    config.updated_by_id = current_user.id

    db.session.commit()

    log_audit_event(
        'monitoring_rule_update',
        resource_type='monitoring_config',
        resource_id=config.id,
        details=f"Updated rules for '{schedule.class_name}': Full PPE={is_full_ppe}, Phones={config.monitor_phones}, Laptops={config.monitor_laptops}"
    )

    flash(f'Safety triggers for "{schedule.class_name}" updated.', 'success')
    return redirect(url_for('classes.monitoring_list'))


# ───────────────────────── Ongoing Classes (Redirect) ─────────────────────────

@classes_bp.route('/classes/ongoing')
@role_required(ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR)
def classes_ongoing():
    """Redirect to consolidated Class Schedule active session tab."""
    return redirect(url_for('classes.classes_list', tab='active'))
