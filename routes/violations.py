"""
Violations Blueprint — violation listing, acknowledgement, and review workflow.
"""
import io
import csv
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, cast
from flask import Blueprint, render_template, request, redirect, url_for, flash, Response
from flask_login import login_required, current_user

from database import db, ViolationLog, ClassSchedule, EmergencyAlert
from safety_engine import get_current_ongoing_class
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM
from audit_utils import log_audit_event
from time_utils import get_myt_now

violations_bp = Blueprint('violations', __name__)


@violations_bp.route('/violations')
@role_required(ROLE_ADMIN, ROLE_AFM)
def violations_list():
    """
    Violations center with 3 dedicated UX views:
      1. 'current' : Unacknowledged violations for current ongoing class.
      2. 'pending' : Unacknowledged violations from previous sessions.
      3. 'history' : Acknowledged violations archive with audit trail.
    """
    tab = request.args.get('tab', 'current')
    page = request.args.get('page', 1, type=int)
    severity = request.args.get('severity', '').strip().lower()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    date_range = request.args.get('date_range', '').strip().lower()
    incident_filter = request.args.get('incident_filter', 'all').strip().lower()
    if incident_filter not in ('all', 'ppe', 'contraband', 'hazard'):
        incident_filter = 'all'

    # Resolve date_range preset if provided
    now = get_myt_now()
    if date_range == 'today':
        start_date = now.strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == '7d':
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == '30d':
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == 'all':
        start_date = ''
        end_date = ''
    elif not date_range:
        if start_date or end_date:
            date_range = 'custom'
        else:
            date_range = '7d'
            start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')

    ongoing = get_current_ongoing_class(db.session, ClassSchedule)
    ongoing_id = ongoing.id if ongoing else None

    # Base counts for tab navigation badges (exclude false positives)
    fp_filter = ViolationLog.review_status != 'false_positive'

    if ongoing_id:
        current_count = ViolationLog.query.filter_by(
            is_acknowledged=False, class_schedule_id=ongoing_id
        ).filter(fp_filter).count()  # type: ignore[arg-type]
        pending_count = ViolationLog.query.filter_by(
            is_acknowledged=False
        ).filter(
            (ViolationLog.class_schedule_id != ongoing_id) | (
                ViolationLog.class_schedule_id.is_(None)
            )
        ).filter(fp_filter).count()  # type: ignore[arg-type]
    else:
        current_count = ViolationLog.query.filter_by(
            is_acknowledged=False
        ).filter(
            ViolationLog.class_schedule_id.is_(None)
        ).filter(fp_filter).count()  # type: ignore[arg-type]
        pending_count = ViolationLog.query.filter_by(
            is_acknowledged=False
        ).filter(
            ViolationLog.class_schedule_id.isnot(None)
        ).filter(fp_filter).count()  # type: ignore[arg-type]

    history_count = ViolationLog.query.filter_by(
        is_acknowledged=True
    ).count()

    # Query based on selected tab
    q = ViolationLog.query
    if severity:
        q = q.filter_by(severity=severity)

    if incident_filter == 'ppe':
        q = q.filter(
            ~ViolationLog.hazard_type.ilike('%fire%'),
            ~ViolationLog.hazard_type.ilike('%smoke%'),
            ~ViolationLog.hazard_type.ilike('%combust%'),
            ~ViolationLog.hazard_type.ilike('%spill%'),
            ~ViolationLog.hazard_type.ilike('%contraband%'),
            ~ViolationLog.hazard_type.ilike('%phone%'),
            ~ViolationLog.hazard_type.ilike('%laptop%'),
        )
    elif incident_filter == 'contraband':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%contraband%') |
            ViolationLog.hazard_type.ilike('%phone%') |
            ViolationLog.hazard_type.ilike('%laptop%')
        )
    elif incident_filter == 'hazard':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%fire%') |
            ViolationLog.hazard_type.ilike('%smoke%') |
            ViolationLog.hazard_type.ilike('%combust%') |
            ViolationLog.hazard_type.ilike('%spill%')
        )

    if tab == 'current':
        if ongoing_id:
            q = q.filter_by(
                is_acknowledged=False,
                class_schedule_id=ongoing_id
            ).filter(fp_filter)  # type: ignore[arg-type]
        else:
            q = q.filter_by(is_acknowledged=False).filter(
                ViolationLog.class_schedule_id.is_(None)
            ).filter(fp_filter)  # type: ignore[arg-type]

    elif tab == 'pending':
        if ongoing_id:
            q = q.filter_by(
                is_acknowledged=False
            ).filter(
                (ViolationLog.class_schedule_id != ongoing_id) | (
                    ViolationLog.class_schedule_id.is_(None)
                )
            ).filter(fp_filter)  # type: ignore[arg-type]
        else:
            q = q.filter_by(is_acknowledged=False).filter(
                ViolationLog.class_schedule_id.isnot(None)
            ).filter(fp_filter)  # type: ignore[arg-type]

    else:
        # History tab ("All Violations" archive)
        tab = 'history'
        q = q.filter_by(is_acknowledged=True)

    # Apply calendar / manual date range filters
    if start_date:
        try:
            s_dt = datetime.strptime(start_date, '%Y-%m-%d')
            q = q.filter(ViolationLog.timestamp >= s_dt)
        except ValueError:
            pass
    if end_date:
        try:
            e_dt = datetime.strptime(end_date, '%Y-%m-%d')
            e_dt = e_dt.replace(hour=23, minute=59, second=59)
            q = q.filter(ViolationLog.timestamp <= e_dt)
        except ValueError:
            pass

    # Absorb emergency hazards into the list view
    emergency_alerts = []
    if incident_filter in ('all', 'hazard'):
        alert_q = EmergencyAlert.query
        if tab in ('current', 'pending'):
            alert_q = alert_q.filter_by(is_acknowledged=False)
        else:
            alert_q = alert_q.filter_by(is_acknowledged=True)
        emergency_alerts = alert_q.order_by(EmergencyAlert.timestamp.desc()).all()

    # Show 10 items per page by default (10 most recent)
    violations = q.order_by(
        ViolationLog.timestamp.desc()
    ).paginate(page=page, per_page=10, error_out=False)

    return render_template(
        'violations.html',
        violations=violations,
        current_severity=severity,
        active_tab=tab,
        ongoing_class=ongoing,
        current_count=current_count,
        pending_count=pending_count,
        history_count=history_count,
        start_date=start_date,
        end_date=end_date,
        date_range=date_range,
        incident_filter=incident_filter,
        emergency_alerts=emergency_alerts,
    )


@violations_bp.route('/violations/export')
@role_required(ROLE_ADMIN, ROLE_AFM)
def export_violations():
    """Export filtered violations to downloadable CSV format (Admin and AFM only)."""
    tab = request.args.get('tab', 'current')
    severity = request.args.get('severity', '').strip().lower()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    date_range = request.args.get('date_range', '').strip().lower()
    incident_filter = request.args.get('incident_filter', 'all').strip().lower()

    now = get_myt_now()
    if date_range == 'today':
        start_date = now.strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == '7d':
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == '30d':
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
    elif date_range == 'all':
        start_date = ''
        end_date = ''

    ongoing = get_current_ongoing_class(db.session, ClassSchedule)
    ongoing_id = ongoing.id if ongoing else None

    q = ViolationLog.query
    if severity:
        q = q.filter_by(severity=severity)

    if incident_filter == 'ppe':
        q = q.filter(
            ~ViolationLog.hazard_type.ilike('%fire%'),
            ~ViolationLog.hazard_type.ilike('%smoke%'),
            ~ViolationLog.hazard_type.ilike('%combust%'),
            ~ViolationLog.hazard_type.ilike('%spill%'),
            ~ViolationLog.hazard_type.ilike('%contraband%'),
            ~ViolationLog.hazard_type.ilike('%phone%'),
            ~ViolationLog.hazard_type.ilike('%laptop%'),
        )
    elif incident_filter == 'contraband':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%contraband%') |
            ViolationLog.hazard_type.ilike('%phone%') |
            ViolationLog.hazard_type.ilike('%laptop%')
        )
    elif incident_filter == 'hazard':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%fire%') |
            ViolationLog.hazard_type.ilike('%smoke%') |
            ViolationLog.hazard_type.ilike('%combust%') |
            ViolationLog.hazard_type.ilike('%spill%')
        )

    if tab == 'current':
        if ongoing_id:
            q = q.filter_by(is_acknowledged=False, class_schedule_id=ongoing_id)
        else:
            q = q.filter_by(is_acknowledged=False).filter(
                ViolationLog.class_schedule_id.is_(None)
            )
    elif tab == 'pending':
        if ongoing_id:
            q = q.filter_by(is_acknowledged=False).filter(
                (ViolationLog.class_schedule_id != ongoing_id) | (
                    ViolationLog.class_schedule_id.is_(None)
                )
            )
        else:
            q = q.filter_by(is_acknowledged=False).filter(
                ViolationLog.class_schedule_id.isnot(None)
            )
    else:
        q = q.filter_by(is_acknowledged=True)

    if start_date:
        try:
            s_dt = datetime.strptime(start_date, '%Y-%m-%d')
            q = q.filter(ViolationLog.timestamp >= s_dt)
        except ValueError:
            pass
    if end_date:
        try:
            e_dt = datetime.strptime(end_date, '%Y-%m-%d')
            e_dt = e_dt.replace(hour=23, minute=59, second=59)
            q = q.filter(ViolationLog.timestamp <= e_dt)
        except ValueError:
            pass

    records = q.order_by(ViolationLog.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Incident ID", "Timestamp (UTC)", "Hazard / Violation Type",
        "Severity", "Camera Source", "Course Code", "Course Name",
        "Lab Room", "Review Status", "Resolution Note", "Evidence Hash (SHA-256)"
    ])

    for r in records:
        cam_code = r.camera.code if r.camera else "CAM-01"
        course_code = r.class_schedule.class_code if r.class_schedule else "None"
        course_name = r.class_schedule.class_name if r.class_schedule else "Unassigned"
        lab_room = r.class_schedule.lab_room if r.class_schedule else (r.camera.location if r.camera else "Lab Area")
        ts_str = r.timestamp.strftime('%d/%m/%y %H:%M:%S') if r.timestamp else ""
        writer.writerow([
            f"VIO-{r.id:05d}", ts_str, r.hazard_type, r.severity.upper(),
            cam_code, course_code, course_name, lab_room,
            r.review_status, r.resolution_note or "", r.evidence_hash or ""
        ])

    csv_data = output.getvalue()
    filename = f"safelabs_violations_{get_myt_now().strftime('%Y%m%d_%H%M%S')}.csv"

    log_audit_event(
        'violations_export_csv',
        resource_type='violation_log',
        details=f"Exported violations CSV: {filename} (Tab: {tab}, Records: {len(records)})"
    )

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@violations_bp.route('/violations/<int:vid>/acknowledge', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def violation_acknowledge(vid):
    """Acknowledge violation with optional review classification (Admin & AFM only)."""
    v = ViolationLog.query.get_or_404(vid)
    note = request.form.get('resolution_note', '').strip()
    origin_tab = request.form.get('tab', 'current')
    review = request.form.get('review_status', 'confirmed')

    # Validate review_status
    if review not in ('confirmed', 'false_positive', 'false_negative'):
        review = 'confirmed'

    now = get_myt_now()
    v.is_acknowledged = True
    v.acknowledged_by_id = current_user.id
    v.acknowledged_at = now
    v.review_status = review
    v.reviewed_by_id = current_user.id
    if note:
        v.resolution_note = note

    # Synchronize corresponding EmergencyAlert if this violation was an emergency hazard
    if v.image_path:
        matching_alert = EmergencyAlert.query.filter_by(image_path=v.image_path).first()
        if matching_alert:
            matching_alert.is_acknowledged = True
            matching_alert.acknowledged_by_id = current_user.id
            matching_alert.acknowledged_at = now
            matching_alert.review_status = review
            matching_alert.reviewed_by_id = current_user.id
            if note:
                matching_alert.resolution_note = note

    db.session.commit()

    log_audit_event(
        'violation_status_update',
        resource_type='violation_log',
        resource_id=v.id,
        details=f"Violation #{v.id} triaged/reviewed as '{review}' by {current_user.username} (Role: {current_user.role})"
    )

    if review == 'false_positive':
        flash(
            f'Violation #{v.id} marked as False Positive and archived.',
            'info',
        )
    elif review == 'false_negative':
        flash(
            f'Violation #{v.id} marked as Borderline and archived.',
            'info',
        )
    else:
        flash(
            f'Violation #{v.id} acknowledged and moved to Violation History.',
            'success',
        )
    return redirect(url_for('violations.violations_list', tab=origin_tab))


@violations_bp.route('/violations/clear_all', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def violations_clear_all():
    """Acknowledge and archive unacknowledged violations (Admin and AFM only)."""
    tab = request.form.get('tab', 'current')
    incident_filter = request.form.get('incident_filter', 'all')
    scope = request.form.get('scope', 'auto')

    q = ViolationLog.query.filter_by(is_acknowledged=False)

    # Filter by incident type if user selected a filter
    if incident_filter == 'ppe':
        q = q.filter(
            ~ViolationLog.hazard_type.ilike('%fire%'),
            ~ViolationLog.hazard_type.ilike('%smoke%'),
            ~ViolationLog.hazard_type.ilike('%combust%'),
            ~ViolationLog.hazard_type.ilike('%spill%'),
            ~ViolationLog.hazard_type.ilike('%contraband%'),
            ~ViolationLog.hazard_type.ilike('%phone%'),
            ~ViolationLog.hazard_type.ilike('%laptop%'),
        )
    elif incident_filter == 'contraband':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%contraband%') |
            ViolationLog.hazard_type.ilike('%phone%') |
            ViolationLog.hazard_type.ilike('%laptop%')
        )
    elif incident_filter == 'hazard':
        q = q.filter(
            ViolationLog.hazard_type.ilike('%fire%') |
            ViolationLog.hazard_type.ilike('%smoke%') |
            ViolationLog.hazard_type.ilike('%combust%') |
            ViolationLog.hazard_type.ilike('%spill%')
        )

    ongoing = get_current_ongoing_class(db.session, ClassSchedule)
    ongoing_id = ongoing.id if ongoing else None

    if scope == 'tab':
        if tab == 'current':
            if ongoing_id:
                q = q.filter_by(class_schedule_id=ongoing_id)
            else:
                q = q.filter(ViolationLog.class_schedule_id.is_(None))
        elif tab == 'pending':
            if ongoing_id:
                q = q.filter(
                    (ViolationLog.class_schedule_id != ongoing_id) | (
                        ViolationLog.class_schedule_id.is_(None)
                    )
                )
            else:
                q = q.filter(ViolationLog.class_schedule_id.isnot(None))

    unack = q.all()
    count = len(unack)
    now = get_myt_now()
    for item in unack:
        item.is_acknowledged = True
        item.acknowledged_by_id = current_user.id
        item.acknowledged_at = now
        item.review_status = 'confirmed'
        item.reviewed_by_id = current_user.id
        if not item.resolution_note:
            item.resolution_note = "Batch cleared via Clear All."

    if incident_filter in ('all', 'hazard'):
        unack_alerts = EmergencyAlert.query.filter_by(is_acknowledged=False).all()
        for a in unack_alerts:
            a.is_acknowledged = True
            a.acknowledged_by_id = current_user.id
            a.acknowledged_at = now
            if not a.resolution_note:
                a.resolution_note = "Batch cleared via Clear All."

    db.session.commit()

    log_audit_event(
        'violations_purge_batch',
        resource_type='violation_log',
        details=f"Batch cleared {count} violations (Filter: {incident_filter}, Scope: {scope})"
    )

    filter_label = {
        'ppe': 'PPE compliance',
        'contraband': 'tabletop contraband',
        'hazard': 'fire & smoke hazard',
    }.get(incident_filter, '')
    lbl_str = f" {filter_label}" if filter_label else ""
    flash(f"Successfully acknowledged and cleared {count}{lbl_str} violations.", "success")
    return redirect(url_for('violations.violations_list', tab=tab, incident_filter=incident_filter))
