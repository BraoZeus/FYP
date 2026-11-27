"""
Authentication Blueprint — login / logout with rate limiting and audit logging.
"""
import re
import time
from datetime import datetime
from collections import defaultdict
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user

from database import db, User
from auth_utils import validate_password_complexity
from audit_utils import (
    log_audit_event,
    ACTION_LOGIN_SUCCESS, ACTION_LOGIN_FAILED, ACTION_LOGOUT,
)
from time_utils import get_myt_now

auth_bp = Blueprint('auth', __name__)


# ---------------------------------------------------------------------------
# In-memory rate limiter for login attempts (5 per minute per IP)
# ---------------------------------------------------------------------------
_login_attempts = defaultdict(list)  # {ip: [timestamp, ...]}
_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 60


def _is_rate_limited(ip: str) -> bool:
    """Check if an IP has exceeded the login attempt rate limit."""
    now = time.time()
    # Prune old attempts outside the window
    _login_attempts[ip] = [
        t for t in _login_attempts[ip] if now - t < _WINDOW_SECONDS
    ]
    return len(_login_attempts[ip]) >= _MAX_ATTEMPTS


def _record_attempt(ip: str):
    """Record a login attempt for the given IP."""
    _login_attempts[ip].append(time.time())


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Sign-in handler for authenticated sessions with rate limiting."""
    if request.method == 'GET' and current_user.is_authenticated:
        logout_user()

    if request.method == 'POST':
        client_ip = request.remote_addr or '0.0.0.0'

        # Rate limit check
        if _is_rate_limited(client_ip):
            flash(
                'Too many login attempts. Please wait 60 seconds before trying again.',
                'error',
            )
            log_audit_event(
                ACTION_LOGIN_FAILED,
                username=request.form.get('username', '').strip().lower(),
                details=f'Rate limited — exceeded {_MAX_ATTEMPTS} attempts/min',
                success=False,
            )
            return redirect(url_for('auth.login'))

        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return redirect(url_for('auth.login'))

        _record_attempt(client_ip)
        user = User.query.filter_by(username=username).first()

        # Check account lockout status
        if user and user.is_locked():
            remaining_seconds = int((user.locked_until - get_myt_now()).total_seconds())
            remaining_minutes = max(1, (remaining_seconds + 59) // 60)
            flash(
                f'Account is temporarily locked due to excessive failed attempts. Try again in {remaining_minutes} minute(s) or contact an administrator.',
                'error'
            )
            log_audit_event(
                ACTION_LOGIN_FAILED,
                username=username,
                user_id=user.id,
                details=f'Blocked by account lockout policy ({remaining_minutes} min remaining)',
                success=False,
            )
            return redirect(url_for('auth.login'))

        if user and user.is_active_account and user.check_password(password):
            # Reset failed attempts upon successful authentication
            user.unlock_account()
            db.session.commit()

            login_user(user, remember=True)

            # Audit: successful login
            log_audit_event(
                ACTION_LOGIN_SUCCESS,
                user_id=user.id,
                username=user.username,
                details='Successful authentication',
            )

            # Mandatory password change enforcement
            if user.must_change_password:
                flash(
                    'Temporary password detected. Please set a new personal password before accessing the portal.',
                    'info',
                )
                return redirect(url_for('auth.change_password'))

            flash(
                f'Welcome back, {user.full_name or user.username}.',
                'success',
            )

            next_page = request.args.get('next')
            if next_page and not next_page.startswith('/'):
                next_page = None
            if user.role == 'academic_coordinator':
                if not next_page or next_page in ('/dashboard', '/'):
                    next_page = url_for('classes.classes_list')
            else:
                if not next_page:
                    next_page = url_for('main.dashboard')
            return redirect(next_page)

        # Handle failed login & increment lockout tracker
        if user and user.is_active_account:
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            if user.failed_login_attempts >= 5:
                user.lock_account(minutes=15)
                db.session.commit()
                log_audit_event(
                    ACTION_LOGIN_FAILED,
                    username=username,
                    user_id=user.id,
                    details='Account locked for 15 minutes after 5 consecutive failed attempts',
                    success=False,
                )
                flash('Account has been locked for 15 minutes due to 5 consecutive failed login attempts.', 'error')
                return redirect(url_for('auth.login'))
            else:
                attempts_left = 5 - user.failed_login_attempts
                db.session.commit()
                flash(f'Invalid credentials. {attempts_left} attempt(s) remaining before account lockout.', 'error')
        else:
            flash('Invalid credentials or account disabled.', 'error')

        # Audit: failed login attempt
        log_audit_event(
            ACTION_LOGIN_FAILED,
            username=username,
            details='Invalid credentials or disabled account',
            success=False,
        )
        return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Terminate authenticated session with audit trail."""
    reason = request.args.get('reason')
    details = 'User logged out'
    if reason == 'inactivity':
        details = 'User automatically logged out due to 15 minutes of inactivity'

    log_audit_event(
        ACTION_LOGOUT,
        details=details,
    )
    logout_user()
    if reason == 'inactivity':
        flash('You have been logged out due to 15 minutes of inactivity for laboratory security.', 'info')
    else:
        flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Mandatory or self-service password update screen with rate limiting."""
    if request.method == 'POST':
        client_ip = request.remote_addr or '0.0.0.0'
        rate_key = f'chg_pw_{client_ip}'
        if _is_rate_limited(rate_key):
            flash('Too many password update attempts. Please wait 60 seconds before trying again.', 'error')
            return render_template('change_password.html')
        _record_attempt(rate_key)

        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Strong password complexity validation
        is_valid, err_msg = validate_password_complexity(new_password)
        if not is_valid:
            flash(err_msg, 'error')
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash('Passwords do not match. Please verify and re-enter.', 'error')
            return render_template('change_password.html')

        if current_user.check_password(new_password):
            flash('New password cannot be identical to your temporary password.', 'error')
            return render_template('change_password.html')

        current_user.set_password(new_password)
        current_user.must_change_password = False
        db.session.commit()

        log_audit_event(
            'password_change',
            user_id=current_user.id,
            username=current_user.username,
            resource_type='user',
            resource_id=current_user.id,
            details='User successfully set a new permanent password',
        )

        flash('Your password has been set successfully. Welcome to SAFELABS!', 'success')

        if current_user.role == 'academic_coordinator':
            return redirect(url_for('classes.classes_list'))
        return redirect(url_for('main.dashboard'))

    return render_template('change_password.html')
