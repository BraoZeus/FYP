"""
Role-Based Access Control (RBAC) and authentication utilities for SAFELABS.
Defines role constants and the central @role_required decorator.
"""
import re
from functools import wraps
from flask import request, jsonify, abort
from flask_login import current_user, login_required

from audit_utils import log_audit_event

ROLE_ADMIN = 'admin'
ROLE_AFM = 'afm'
ROLE_ACADEMIC_COORDINATOR = 'academic_coordinator'

VALID_ROLES = [ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR]


def validate_password_complexity(password: str) -> tuple[bool, str]:
    """
    Validate that a password meets SAFELABS strong complexity requirements:
      - At least 8 characters long
      - At least one uppercase letter (A-Z)
      - At least one lowercase letter (a-z)
      - At least one numeric digit (0-9)
      - At least one special character (!@#$%^&*()-_=+[]{}|;:,.<>/?~`)
    Returns (is_valid: bool, error_message: str)
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number (0-9)."
    if not re.search(r'[!@#$%^&*()_\-+=\[\]{}|;:,.<>?/~`]', password):
        return False, "Password must contain at least one special character (e.g. !@#$%^&*)."
    return True, ""


def role_required(*allowed_roles):
    """
    Restrict endpoint execution to authenticated users with permitted roles.

    If unauthorized:
      - Logs an audit security violation event into AuditLog.
      - Returns HTTP 403 JSON if the request is an API/JSON endpoint.
      - Aborts with HTTP 403 for browser requests (strict server-side blocking).
    """
    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            user_role = getattr(current_user, 'role', None)
            if not current_user.is_authenticated or user_role not in allowed_roles:
                log_audit_event(
                    action='unauthorized_access_attempt',
                    details=(
                        f"Access denied to '{request.path}' for user "
                        f"'{getattr(current_user, 'username', 'anonymous')}' (Role: {user_role}). "
                        f"Required roles: {allowed_roles}"
                    ),
                    success=False,
                )
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Forbidden',
                        'message': 'Insufficient permissions for this action.',
                        'required_roles': list(allowed_roles),
                    }), 403

                abort(403)

            return fn(*args, **kwargs)
        return wrapped
    return decorator


# Backward-compatibility alias
requires_roles = role_required
