"""
Audit logging utilities for SAFELABS security event tracking.

Logs security-sensitive actions: login, logout, failed login, and
evidence access events into the AuditLog database table.
"""
from flask import request
from flask_login import current_user

from database import db, AuditLog


# ---------------------------------------------------------------------------
# Audit Action Constants
# ---------------------------------------------------------------------------

ACTION_LOGIN_SUCCESS = 'login_success'
ACTION_LOGIN_FAILED = 'login_failed'
ACTION_LOGOUT = 'logout'
ACTION_EVIDENCE_VIEW = 'evidence_view'
ACTION_EVIDENCE_INTEGRITY_FAIL = 'evidence_integrity_failure'


def log_audit_event(
    action: str,
    user_id: int | None = None,
    username: str | None = None,
    resource_type: str | None = None,
    resource_id: int | None = None,
    details: str | None = None,
    success: bool = True,
    event_type: str | None = None,
):
    """
    Record a security audit event in the database.

    If user_id / username are not provided, attempts to infer from
    the currently authenticated user (if any).

    Parameters:
        action:        Action constant (e.g., ACTION_LOGIN_SUCCESS).
        user_id:       ID of the user performing the action.
        username:      Username string for the event.
        resource_type: Type of resource accessed (e.g., 'violation', 'alert').
        resource_id:   Primary key of the accessed resource.
        details:       Free-text description of the event.
        success:       Whether the action succeeded.
        event_type:    Log severity: 'INFO', 'WARNING', 'ERROR' (auto-derived if omitted).
    """
    # Auto-populate user info from session if available
    if user_id is None and hasattr(current_user, 'id') and current_user.is_authenticated:
        user_id = current_user.id
    if username is None and hasattr(current_user, 'username') and current_user.is_authenticated:
        username = current_user.username

    # Extract request context
    ip_address = None
    user_agent = None
    try:
        ip_address = request.remote_addr
        user_agent = str(request.user_agent)[:255] if request.user_agent else None
    except RuntimeError:
        # Outside of request context (e.g., CLI usage)
        pass

    entry = AuditLog(
        action=action,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        success=success,
        event_type=event_type,
    )
    db.session.add(entry)
    db.session.commit()
    return entry
