import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app import app
from database import db, User, ClassSchedule, MonitoringConfig
from auth_utils import ROLE_ADMIN, ROLE_AFM, ROLE_ACADEMIC_COORDINATOR, VALID_ROLES


@pytest.fixture
def app_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        # Ensure test users exist with correct roles
        for role, uname in [
            (ROLE_ADMIN, 'admin'),
            (ROLE_AFM, 'afm'),
            (ROLE_ACADEMIC_COORDINATOR, 'academic_coordinator')
        ]:
            u = User.query.filter_by(username=uname).first()
            if not u:
                u = User(username=uname, role=role, full_name=uname.replace('_', ' ').title(), must_change_password=False)
                u.set_password(uname)
                db.session.add(u)
            else:
                u.role = role
                u.must_change_password = False
                u.set_password(uname)
        
        admin_user = User.query.filter_by(role=ROLE_ADMIN).first()
        # Ensure a test class schedule exists for testing CRUD & rules
        test_class = ClassSchedule.query.filter_by(class_code='TEST101').first()
        if not test_class:
            test_class = ClassSchedule(
                class_name='Test Biology',
                class_code='TEST101',
                lab_room='Lab A',
                instructor='Dr. Test',
                day_of_week='Monday',
                start_time=ClassSchedule.start_time.type.python_type(9, 0),
                end_time=ClassSchedule.end_time.type.python_type(11, 0),
                semester='Spring 2026',
                is_active=True,
                created_by_id=admin_user.id if admin_user else 1,
            )
            db.session.add(test_class)
            db.session.flush()
            m_cfg = MonitoringConfig(
                class_schedule_id=test_class.id,
                monitor_full_ppe=True,
                monitor_phones=False,
                monitor_laptops=False,
                updated_by_id=admin_user.id if admin_user else 1,
            )
            db.session.add(m_cfg)
        db.session.commit()
    return app.test_client()


def test_valid_roles_and_db_migration():
    """Verify valid role constants and that lab_assistant is migrated to academic_coordinator."""
    assert VALID_ROLES == ['admin', 'afm', 'academic_coordinator']
    with app.app_context():
        # lab_assistant role should not exist in database
        lab_users = User.query.filter_by(role='lab_assistant').all()
        assert len(lab_users) == 0

        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
        assert coord is not None
        assert coord.role_display == 'Academic Coordinator'


def test_login_and_index_redirects(app_client):
    """Verify login and index redirects for academic_coordinator vs admin/afm."""
    # 1. Academic coordinator index redirect
    with app.app_context():
        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True
    res = app_client.get('/')
    assert res.status_code == 302
    assert '/class_schedule' in res.headers['Location'] or '/classes' in res.headers['Location']

    # 2. Admin index redirect
    with app.app_context():
        admin = User.query.filter_by(role=ROLE_ADMIN).first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(admin.id)
        sess['_fresh'] = True
    res = app_client.get('/')
    assert res.status_code == 302
    assert '/dashboard' in res.headers['Location']

    # 3. AFM index redirect
    with app.app_context():
        afm = User.query.filter_by(role=ROLE_AFM).first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(afm.id)
        sess['_fresh'] = True
    res = app_client.get('/')
    assert res.status_code == 302
    assert '/dashboard' in res.headers['Location']


def test_academic_coordinator_strict_route_blocking(app_client):
    """Verify academic_coordinator receives HTTP 403 on all unauthorized routes."""
    with app.app_context():
        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
        test_class = ClassSchedule.query.filter_by(class_code='TEST101').first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True

    # Dashboard is blocked (403)
    res = app_client.get('/dashboard')
    assert res.status_code == 403

    # Live feeds routes are blocked (403)
    for route in ['/live_feeds', '/live_feed', '/video_feed', '/video_feed/1']:
        res = app_client.get(route)
        assert res.status_code == 403, f"Expected 403 for {route}, got {res.status_code}"

    res = app_client.post('/capture_evidence')
    assert res.status_code == 403

    # Violations & Evidence routes are blocked (403)
    for route in ['/violations', '/violations/export', '/alerts', '/evidence/test.jpg']:
        res = app_client.get(route)
        assert res.status_code == 403, f"Expected 403 for {route}, got {res.status_code}"

    res = app_client.post('/violations/1/acknowledge')
    assert res.status_code == 403
    res = app_client.post('/violations/clear_all')
    assert res.status_code == 403

    # User management is blocked (403)
    res = app_client.get('/users')
    assert res.status_code == 403

    # Detection rules POST (mutating safety rules) is blocked (403)
    res = app_client.post(f'/monitoring/{test_class.id}', data={'monitor_full_ppe': 'on'})
    assert res.status_code == 403
    res = app_client.post(f'/detection_rules/{test_class.id}', data={'monitor_full_ppe': 'on'})
    assert res.status_code == 403

    # BUT Allowed routes work:
    # Class schedule GET allowed (200)
    res = app_client.get('/class_schedule')
    assert res.status_code == 200
    res = app_client.get('/classes')
    assert res.status_code == 200

    # Detection rules GET allowed (200)
    res = app_client.get('/detection_rules')
    assert res.status_code == 200
    res = app_client.get('/monitoring')
    assert res.status_code == 200


def test_afm_schedule_mutation_blocking(app_client):
    """Verify AFM is blocked from mutating timetable schedules (POST/PUT/DELETE) with HTTP 403."""
    with app.app_context():
        afm = User.query.filter_by(role=ROLE_AFM).first()
        test_class = ClassSchedule.query.filter_by(class_code='TEST101').first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(afm.id)
        sess['_fresh'] = True

    # AFM can view class schedule (200)
    res = app_client.get('/class_schedule')
    assert res.status_code == 200

    # AFM blocked from GET create form (403)
    res = app_client.get('/classes/create')
    assert res.status_code == 403

    # AFM blocked from POST create class (403)
    res = app_client.post('/classes/create', data={'class_name': 'Unauthorized'})
    assert res.status_code == 403

    # AFM blocked from GET/POST edit class (403)
    res = app_client.get(f'/classes/{test_class.id}/edit')
    assert res.status_code == 403
    res = app_client.post(f'/classes/{test_class.id}/edit', data={'class_name': 'Changed'})
    assert res.status_code == 403

    # AFM blocked from DELETE class (403)
    res = app_client.post(f'/classes/{test_class.id}/delete')
    assert res.status_code == 403

    # AFM blocked from User management (403)
    res = app_client.get('/users')
    assert res.status_code == 403


def test_academic_coordinator_schedule_crud_allowed(app_client):
    """Verify academic_coordinator has full CRUD access to class schedules."""
    with app.app_context():
        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
        test_class = ClassSchedule.query.filter_by(class_code='TEST101').first()
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True

    # GET create allowed (200)
    res = app_client.get('/classes/create')
    assert res.status_code == 200

    # GET edit allowed (200)
    res = app_client.get(f'/classes/{test_class.id}/edit')
    assert res.status_code == 200


def test_sidebar_stripping_and_badges(app_client):
    """Verify complete DOM stripping of unauthorized navigation links and audio toggle."""
    with app.app_context():
        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
        afm = User.query.filter_by(role=ROLE_AFM).first()
        admin = User.query.filter_by(role=ROLE_ADMIN).first()

    # 1. Academic Coordinator Sidebar
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True
    res = app_client.get('/class_schedule')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "Class Schedule" in html
    assert "Detection Rules" in html
    assert "Dashboard" not in html
    assert "Live Feeds" not in html
    assert "Violations" not in html
    assert "User Management" not in html
    assert "audioToggleBtn" not in html
    assert "ACADEMIC COORD" in html

    # 2. AFM Sidebar
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(afm.id)
        sess['_fresh'] = True
    res = app_client.get('/dashboard')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "Dashboard" in html
    assert "Live Feeds" in html
    assert "Violations" in html
    assert "Class Schedule" in html
    assert "Detection Rules" in html
    assert "User Management" not in html
    assert "audioToggleBtn" in html
    assert "AFM" in html

    # 3. Admin Sidebar
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(admin.id)
        sess['_fresh'] = True
    res = app_client.get('/dashboard')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "Dashboard" in html
    assert "Live Feeds" in html
    assert "Violations" in html
    assert "Class Schedule" in html
    assert "Detection Rules" in html
    assert "User Management" in html
    assert "audioToggleBtn" in html
    assert "ADMIN" in html


def test_template_controls_and_read_only_views(app_client):
    """Verify page-level conditionals for AFM and Academic Coordinator."""
    with app.app_context():
        coord = User.query.filter_by(role=ROLE_ACADEMIC_COORDINATOR).first()
        afm = User.query.filter_by(role=ROLE_AFM).first()
        admin = User.query.filter_by(role=ROLE_ADMIN).first()

    # 1. Class Schedule: AFM has NO "+ Add Class" and NO Actions/Edit/Delete
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(afm.id)
        sess['_fresh'] = True
    res = app_client.get('/class_schedule?tab=timetable')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "Add Class" not in html
    assert ">Actions</div>" not in html

    # Academic coordinator HAS "+ Add Class" and Actions/Edit/Delete
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True
    res = app_client.get('/class_schedule?tab=timetable')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "Add Class" in html
    assert ">Actions</div>" in html

    # 2. Detection Rules: Academic coordinator has disabled switches and NO Save button
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(coord.id)
        sess['_fresh'] = True
    res = app_client.get('/detection_rules')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "disabled" in html
    assert 'type="submit"' not in html
    assert "Read-Only" in html

    # AFM HAS enabled Save button on Detection Rules
    with app_client.session_transaction() as sess:
        sess['_user_id'] = str(afm.id)
        sess['_fresh'] = True
    res = app_client.get('/detection_rules')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert 'type="submit"' in html
    assert "Save" in html
