"""
Admin user management Blueprint.
"""
import re
import time
from collections import defaultdict
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from database import db, User, AuditLog
from auth_utils import role_required, VALID_ROLES, ROLE_ADMIN, validate_password_complexity
from audit_utils import log_audit_event

admin_bp = Blueprint('admin', __name__)

_admin_action_attempts = defaultdict(list)
_MAX_ADMIN_ATTEMPTS = 5
_WINDOW_SECONDS = 60

def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    _admin_action_attempts[ip] = [t for t in _admin_action_attempts[ip] if now - t < _WINDOW_SECONDS]
    return len(_admin_action_attempts[ip]) >= _MAX_ADMIN_ATTEMPTS

def _record_attempt(ip: str):
    _admin_action_attempts[ip].append(time.time())


@admin_bp.route('/admin/users')
@admin_bp.route('/users')
@role_required(ROLE_ADMIN)
def admin_users():
    """List all user accounts (Strictly admin only)."""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)


@admin_bp.route('/admin/users/create', methods=['GET', 'POST'])
@admin_bp.route('/users/create', methods=['GET', 'POST'])
@role_required(ROLE_ADMIN)
def admin_create_user():
    """Register a new staff or admin user account (Strictly admin only)."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        role = request.form.get('role', '')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()

        errors = _validate_user_form(username, password, confirm, role, email)
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template(
                'admin_user_form.html', mode='create',
                form_data=request.form, roles=VALID_ROLES,
            )

        user = User(
            username=username, role=role,
            full_name=full_name or None, email=email or None,
            must_change_password=True,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        log_audit_event(
            'user_create',
            resource_type='user',
            resource_id=user.id,
            details=f"Created user '{username}' with role '{role}' (must_change_password=True)"
        )

        flash(f'User "{username}" created with temporary password. They will be prompted to set a new password on their first login.', 'success')
        return redirect(url_for('admin.admin_users'))

    return render_template(
        'admin_user_form.html', mode='create',
        form_data={}, roles=VALID_ROLES,
    )


@admin_bp.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@role_required(ROLE_ADMIN)
def admin_edit_user(user_id):
    """Update user credentials, role, or active status (Strictly admin only)."""
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        role = request.form.get('role', '')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        new_pw = request.form.get('password', '')
        is_active = request.form.get('is_active') == 'on'

        errors = []
        if role not in VALID_ROLES:
            errors.append('Invalid role.')
        if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            errors.append('Invalid email format.')
        if new_pw:
            is_valid, msg = validate_password_complexity(new_pw)
            if not is_valid:
                errors.append(msg)

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template(
                'admin_user_form.html', mode='edit',
                user=user, roles=VALID_ROLES, form_data=request.form,
            )

        user.role = role
        user.full_name = full_name or None
        user.email = email or None
        user.is_active_account = is_active
        if new_pw:
            user.set_password(new_pw)
            user.must_change_password = True
        elif 'must_change_password' in request.form:
            user.must_change_password = request.form.get('must_change_password') == 'on'

        db.session.commit()

        log_audit_event(
            'user_edit',
            resource_type='user',
            resource_id=user.id,
            details=f"Updated user '{user.username}' (Role: {user.role}, Active: {user.is_active_account}, MustChangePW: {user.must_change_password})"
        )

        flash(f'User "{user.username}" updated.', 'success')
        return redirect(url_for('admin.admin_users'))

    return render_template(
        'admin_user_form.html', mode='edit',
        user=user, roles=VALID_ROLES, form_data={},
    )


@admin_bp.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@role_required(ROLE_ADMIN)
def admin_delete_user(user_id):
    """Delete a user account (Strictly admin only)."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("You cannot delete your own account.", 'error')
        return redirect(url_for('admin.admin_users'))

    username = user.username
    db.session.delete(user)
    db.session.commit()

    log_audit_event(
        'user_delete',
        resource_type='user',
        resource_id=user_id,
        details=f"Deleted user account '{username}'"
    )

    flash(f'User "{username}" deleted.', 'success')
    return redirect(url_for('admin.admin_users'))


@admin_bp.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@admin_bp.route('/users/<int:user_id>/reset_password', methods=['POST'])
@role_required(ROLE_ADMIN)
def admin_reset_password(user_id):
    """Reset a user's password with a temporary password (Strictly admin only)."""
    client_ip = request.remote_addr or '0.0.0.0'
    if _is_rate_limited(client_ip):
        flash('Too many password reset requests. Please wait 60 seconds before trying again.', 'error')
        return redirect(url_for('admin.admin_users'))
    _record_attempt(client_ip)

    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot reset your own password via this administrative flow.', 'error')
        return redirect(url_for('admin.admin_users'))

    temp_password = request.form.get('temp_password', '').strip()
    if temp_password:
        is_valid, msg = validate_password_complexity(temp_password)
        if not is_valid:
            flash(f'Temporary password does not meet complexity policy: {msg}', 'error')
            return redirect(url_for('admin.admin_users'))
    else:
        import secrets
        import string
        chars = string.ascii_letters + string.digits
        temp_password = 'Safe!' + ''.join(secrets.choice(chars) for _ in range(8))

    user.set_password(temp_password)
    user.must_change_password = True
    user.unlock_account()
    db.session.commit()

    log_audit_event(
        'password_reset_admin',
        resource_type='user',
        resource_id=user.id,
        details=f"Admin issued temporary password reset for '{user.username}'"
    )

    flash(
        f'Password for "{user.username}" reset successfully. Temporary password: "{temp_password}". The user will be required to change it upon next login.',
        'success'
    )
    return redirect(url_for('admin.admin_users'))


@admin_bp.route('/admin/users/<int:user_id>/unlock', methods=['POST'])
@admin_bp.route('/users/<int:user_id>/unlock', methods=['POST'])
@role_required(ROLE_ADMIN)
def admin_unlock_user(user_id):
    """Clear failed login attempts and unlock user account (Admin only)."""
    user = User.query.get_or_404(user_id)
    user.unlock_account()
    db.session.commit()

    log_audit_event(
        'user_unlock',
        resource_type='user',
        resource_id=user.id,
        details=f"Admin unlocked account for '{user.username}'"
    )
    flash(f'Account for "{user.username}" has been unlocked.', 'success')
    return redirect(url_for('admin.admin_users'))


def _validate_user_form(username, password, confirm, role, email):
    """Validate user input fields."""
    errors = []
    if not username or len(username) < 3:
        errors.append('Username must be ≥ 3 characters.')
    if username and not re.match(r'^[a-z0-9_]+$', username):
        errors.append(
            'Username may only contain lowercase letters, numbers, and _'
        )
    is_valid, msg = validate_password_complexity(password)
    if not is_valid:
        errors.append(msg)
    if password != confirm:
        errors.append('Passwords do not match.')
    if role not in VALID_ROLES:
        errors.append('Invalid role selected.')
    if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        errors.append('Invalid email format.')
    return errors


@admin_bp.route('/admin/database_logs')
@admin_bp.route('/database_logs')
@role_required(ROLE_ADMIN)
def database_logs():
    """
    Database Logging Table view (Strictly Admin only).
    Displays immutable system events, audit trails, and telemetry.
    """
    page = request.args.get('page', 1, type=int)
    event_type = request.args.get('event_type', '').strip().upper()
    search = request.args.get('search', '').strip()
    export = request.args.get('export', '')

    query = AuditLog.query

    if event_type and event_type in ['INFO', 'WARNING', 'ERROR']:
        query = query.filter(AuditLog.event_type == event_type)

    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            db.or_(
                AuditLog.details.ilike(search_filter),
                AuditLog.action.ilike(search_filter),
                AuditLog.username.ilike(search_filter)
            )
        )

    if export == 'csv':
        logs = query.order_by(AuditLog.id.desc()).all()
        import io, csv
        from flask import Response
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Timestamp (MYT)', 'Event Type', 'Description', 'User', 'Status'])
        for l in logs:
            writer.writerow([
                f"#{l.id:04d}",
                l.timestamp.strftime('%d/%m/%Y %H:%M:%S') if l.timestamp else '',
                l.event_type or 'INFO',
                l.details or l.action,
                l.username or 'system',
                'Logged'
            ])
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=safelabs_database_logs.csv'}
        )

    pagination = query.order_by(AuditLog.id.desc()).paginate(
        page=page, per_page=15, error_out=False
    )

    total_logs = AuditLog.query.count()
    info_count = AuditLog.query.filter_by(event_type='INFO').count()
    warning_count = AuditLog.query.filter_by(event_type='WARNING').count()
    error_count = AuditLog.query.filter_by(event_type='ERROR').count()

    return render_template(
        'database_logs.html',
        logs=pagination.items,
        pagination=pagination,
        current_event_type=event_type,
        search_query=search,
        total_logs=total_logs,
        info_count=info_count,
        warning_count=warning_count,
        error_count=error_count,
    )
