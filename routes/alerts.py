"""
Emergency Alerts Blueprint (AFM + Admin Triage).
"""
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from database import db, EmergencyAlert, ViolationLog
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM
from audit_utils import log_audit_event
from time_utils import get_myt_now

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/alerts')
@role_required(ROLE_ADMIN, ROLE_AFM)
def alerts_list():
    """Redirect to merged Violations view filtering on Fire & Smoke Hazards."""
    return redirect(url_for('violations.violations_list', incident_filter='hazard'))


@alerts_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def alert_acknowledge(alert_id):
    """Acknowledge an emergency alert with review status and optional resolution note."""
    alert = EmergencyAlert.query.get_or_404(alert_id)
    note = request.form.get('resolution_note', '').strip()
    review = request.form.get('review_status', 'confirmed')

    if review not in ('confirmed', 'false_positive', 'false_negative'):
        review = 'confirmed'

    now = get_myt_now()
    alert.is_acknowledged = True
    alert.acknowledged_by_id = current_user.id
    alert.acknowledged_at = now
    alert.review_status = review
    alert.reviewed_by_id = current_user.id
    if note:
        alert.resolution_note = note

    # Synchronize corresponding ViolationLog record if present
    if alert.image_path:
        matching_v = ViolationLog.query.filter_by(image_path=alert.image_path).first()
        if matching_v:
            matching_v.is_acknowledged = True
            matching_v.acknowledged_by_id = current_user.id
            matching_v.acknowledged_at = now
            matching_v.review_status = review
            matching_v.reviewed_by_id = current_user.id
            if note:
                matching_v.resolution_note = note

    db.session.commit()

    log_audit_event(
        'alert_status_update',
        resource_type='emergency_alert',
        resource_id=alert.id,
        details=f"Emergency Alert #{alert.id} triaged/reviewed as '{review}' by {current_user.username} (Role: {current_user.role})"
    )

    if review == 'false_positive':
        flash(f'Emergency hazard #{alert.id} marked as False Positive and resolved.', 'info')
    elif review == 'false_negative':
        flash(f'Emergency hazard #{alert.id} marked as Borderline and resolved.', 'info')
    else:
        flash(f'Emergency hazard #{alert.id} confirmed and marked resolved.', 'success')

    return_to = request.form.get('return_to') or url_for('violations.violations_list', incident_filter='hazard')
    return redirect(return_to)


@alerts_bp.route('/alerts/clear_all', methods=['POST'])
@role_required(ROLE_ADMIN, ROLE_AFM)
def alert_clear_all():
    """Acknowledge and clear all unacknowledged emergency alerts (Admin and AFM only)."""
    unack = EmergencyAlert.query.filter_by(is_acknowledged=False).all()
    count = len(unack)
    now = get_myt_now()
    for a in unack:
        a.is_acknowledged = True
        a.acknowledged_by_id = current_user.id
        a.acknowledged_at = now
        if not a.resolution_note:
            a.resolution_note = "Batch acknowledged via Clear All Alerts."
    db.session.commit()

    log_audit_event(
        'alerts_purge_batch',
        resource_type='emergency_alert',
        details=f"Batch cleared {count} emergency alerts"
    )

    flash(f"All {count} emergency hazard alerts cleared and acknowledged.", 'success')
    return_to = request.form.get('return_to') or request.referrer or url_for('violations.violations_list')
    return redirect(return_to)
