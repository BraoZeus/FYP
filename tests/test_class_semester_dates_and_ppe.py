"""
Unit & Integration Test: Class Semester Dates, Day-Mismatch Intelligence, & Safety Trigger Redesign
Validates:
  1. ClassSchedule start_date & end_date attributes and semester boundary logic.
  2. ClassSchedule.is_ongoing evaluating within and outside [start_date, end_date].
  3. safety_engine.get_current_ongoing_class() respects semester date bounds.
  4. Day-of-week mismatch intelligence logic (e.g. Wednesday class with Monday semester start).
  5. MonitoringConfig unified Full PPE, Laptop, Phone rules (goggles disabled).
  6. Class creation/editing routes with day mismatch validation and confirmation flag.
"""
import os
import sys
from datetime import date, time, datetime, timedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app import app
from database import db, ClassSchedule, MonitoringConfig, User
from safety_engine import get_current_ongoing_class, is_monitoring_enabled_for_hazard


def test_class_schedule_date_bounds():
    with app.app_context():
        admin = User.query.filter_by(role="admin").first()
        if not admin:
            admin = User(
                username="admin_test",
                email="admin_test@safelabs.local",
                role="admin",
                password_hash="dummy",
                is_active=True,
            )
            db.session.add(admin)
            db.session.commit()

        today = date.today()
        current_day_name = today.strftime("%A")

        # Create schedule with semester active today
        sched_active = ClassSchedule(
            class_name="Active Organic Chem Lab",
            class_code="CHM-101",
            semester="Fall 2026",
            instructor="Dr. Test",
            lab_room="Lab 101",
            day_of_week=current_day_name,
            start_date=today - timedelta(days=7),
            end_date=today + timedelta(days=30),
            start_time=time(0, 0),
            end_time=time(23, 59),
            is_active=True,
            created_by_id=admin.id,
        )
        db.session.add(sched_active)

        # Create schedule with semester ended yesterday
        sched_expired = ClassSchedule(
            class_name="Past Organic Chem Lab",
            class_code="CHM-102",
            semester="Fall 2026",
            instructor="Dr. Test",
            lab_room="Lab 102",
            day_of_week=current_day_name,
            start_date=today - timedelta(days=60),
            end_date=today - timedelta(days=1),
            start_time=time(0, 0),
            end_time=time(23, 59),
            is_active=True,
            created_by_id=admin.id,
        )
        db.session.add(sched_expired)

        # Create schedule with semester starting in the future
        sched_future = ClassSchedule(
            class_name="Future Organic Chem Lab",
            class_code="CHM-103",
            semester="Spring 2027",
            instructor="Dr. Test",
            lab_room="Lab 103",
            day_of_week=current_day_name,
            start_date=today + timedelta(days=5),
            end_date=today + timedelta(days=90),
            start_time=time(0, 0),
            end_time=time(23, 59),
            is_active=True,
            created_by_id=admin.id,
        )
        db.session.add(sched_future)
        db.session.commit()

        try:
            assert sched_active.is_ongoing is True, "Active date range class should be ongoing"
            assert sched_expired.is_ongoing is False, "Expired semester class must NOT be ongoing"
            assert sched_future.is_ongoing is False, "Future semester class must NOT be ongoing"

            # Check engine ongoing class resolution
            resolved = get_current_ongoing_class(db.session, ClassSchedule)
            assert resolved is not None, "Engine should find ongoing class"
            assert resolved.class_code == "CHM-101"
            print("[PASS] ClassSchedule semester date window verification passed.")
        finally:
            db.session.delete(sched_active)
            db.session.delete(sched_expired)
            db.session.delete(sched_future)
            db.session.commit()


def test_safety_hazard_monitoring_rules():
    with app.app_context():
        admin = User.query.filter_by(role="admin").first()
        # Setup class schedule with MonitoringConfig
        sched = ClassSchedule(
            class_name="Safety Rule Test Lab",
            class_code="SAF-201",
            semester="Fall 2026",
            instructor="Prof. Safety",
            lab_room="Lab 201",
            day_of_week="Monday",
            start_time=time(9, 0),
            end_time=time(11, 0),
            is_active=True,
            created_by_id=admin.id if admin else 1,
        )
        db.session.add(sched)
        db.session.flush()

        config = MonitoringConfig(
            class_schedule_id=sched.id,
            monitor_full_ppe=True,
            monitor_face_mask=True,
            monitor_lab_coat=True,
            monitor_goggles=False,
            monitor_laptops=True,
            monitor_phones=False,
            updated_by_id=admin.id if admin else 1,
        )
        db.session.add(config)
        setattr(sched, 'monitoring', config)
        db.session.commit()

        try:
            # 1. Full PPE hazards (mask, lab coat, gloves, general PPE)
            assert is_monitoring_enabled_for_hazard(sched, "missing_mask") is True
            assert is_monitoring_enabled_for_hazard(sched, "missing_lab_coat") is True
            assert is_monitoring_enabled_for_hazard(sched, "missing_gloves") is True
            assert is_monitoring_enabled_for_hazard(sched, "ppe_violation") is True

            # 2. Goggles must ALWAYS return False (removed from detection)
            assert is_monitoring_enabled_for_hazard(sched, "missing_goggles") is False
            assert is_monitoring_enabled_for_hazard(sched, "goggles") is False

            # 3. Laptops should be enabled
            assert is_monitoring_enabled_for_hazard(sched, "unauthorized_laptop") is True
            assert is_monitoring_enabled_for_hazard(sched, "laptop") is True

            # 4. Phones should be disabled (we set monitor_phones=False)
            assert is_monitoring_enabled_for_hazard(sched, "unauthorized_phone") is False
            assert is_monitoring_enabled_for_hazard(sched, "cell phone") is False

            # 5. Disable full PPE and test
            config.monitor_full_ppe = False
            config.monitor_face_mask = False
            config.monitor_lab_coat = False
            db.session.commit()

            assert is_monitoring_enabled_for_hazard(sched, "missing_mask") is False
            assert is_monitoring_enabled_for_hazard(sched, "missing_lab_coat") is False
            assert is_monitoring_enabled_for_hazard(sched, "missing_gloves") is False
            print("[PASS] Full PPE, Laptop, Phone & Goggles removal hazard gates passed.")
        finally:
            db.session.delete(config)
            db.session.delete(sched)
            db.session.commit()


def test_class_form_day_mismatch_intelligence():
    prev_csrf = app.config.get('WTF_CSRF_ENABLED', True)
    app.config['WTF_CSRF_ENABLED'] = False
    try:
        with app.test_client() as client:
            with app.app_context():
                # Cleanup any prior test run leftover
                old_classes = ClassSchedule.query.filter_by(class_code="BIO-301").all()
                for o in old_classes:
                    if o.monitoring:
                        db.session.delete(o.monitoring)
                    db.session.delete(o)
                db.session.commit()

                # Get admin user for login session
                admin = User.query.filter_by(role="admin").first()
                if not admin:
                    admin = User(
                        username="admin_test",
                        email="admin_test@safelabs.local",
                        role="admin",
                        password_hash="dummy",
                        is_active=True,
                    )
                    db.session.add(admin)
                    db.session.commit()

            # Login session
            with client.session_transaction() as sess:
                sess['_user_id'] = str(admin.id)
                sess['_fresh'] = True

            # Test case: Class on Wednesday, but start_date is a Monday (e.g., 2026-09-07 is Monday, Wednesday is 2026-09-09)
            # Without confirmation: system should warn and re-render form
            post_data_unconfirmed = {
                "class_name": "Biochemistry Lab",
                "class_code": "BIO-301",
                "semester": "Fall 2026",
                "instructor": "Dr. Watson",
                "lab_room": "Lab B2",
                "day_of_week": "Wednesday",
                "start_date": "2026-09-07",  # Monday!
                "end_date": "2026-12-18",
                "start_time": "14:00",
                "end_time": "16:00",
                "monitor_full_ppe": "on",
                "monitor_laptops": "on",
                "confirm_day_mismatch": "0",
            }

            resp = client.post("/classes/create", data=post_data_unconfirmed, follow_redirects=True)
            assert resp.status_code == 200
            # Should prompt user about mismatch
            assert b"Date & Day Notice" in resp.data or b"Schedule Verification" in resp.data or b"Monday" in resp.data

            # Verify class was NOT created yet
            with app.app_context():
                created = ClassSchedule.query.filter_by(class_code="BIO-301").first()
                assert created is None, "Unconfirmed mismatched class should not be persisted"

            # Now submit with confirm_day_mismatch="1"
            post_data_confirmed = dict(post_data_unconfirmed)
            post_data_confirmed["confirm_day_mismatch"] = "1"

            resp2 = client.post("/classes/create", data=post_data_confirmed, follow_redirects=True)
            assert resp2.status_code == 200

            # Verify class WAS created and attributes stored properly
            with app.app_context():
                created = ClassSchedule.query.filter_by(class_code="BIO-301").first()
                assert created is not None, "Confirmed class should be created"
                assert created.day_of_week == "Wednesday"
                assert created.start_date == date(2026, 9, 7)
                assert created.end_date == date(2026, 12, 18)
                assert created.monitoring is not None
                assert created.monitoring.monitor_full_ppe is True
                assert created.monitoring.monitor_laptops is True
                assert created.monitoring.monitor_phones is False
                assert created.monitoring.monitor_goggles is False

                # Clean up
                db.session.delete(created.monitoring)
                db.session.delete(created)
                db.session.commit()

            print("[PASS] Day-mismatch intelligence prompt and confirmation flow verified.")
    finally:
        app.config['WTF_CSRF_ENABLED'] = prev_csrf


if __name__ == "__main__":
    test_class_schedule_date_bounds()
    test_safety_hazard_monitoring_rules()
    test_class_form_day_mismatch_intelligence()
    print("\nALL SEMESTER DATE & SAFETY TRIGGER TESTS PASSED!")
