import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
from app import app
from database import db, User, EmergencyAlert, ViolationLog

def run_audio_tests():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        # Step 1: Login
        print("[1] Authenticating as admin...")
        login_page = client.get('/login')
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_page.data.decode('utf-8'))
        token = csrf_match.group(1) if csrf_match else ''
        res = client.post('/login', data={'username': 'admin', 'password': 'admin', 'csrf_token': token}, follow_redirects=True)
        assert res.status_code == 200, f"Login failed: {res.status_code}"
        print("  ✓ Authenticated successfully")

        # Step 2: Check static audio_alerts.js script
        print("\n[2] Checking static/js/audio_alerts.js...")
        js_res = client.get('/static/js/audio_alerts.js')
        assert js_res.status_code == 200, f"Failed to fetch audio_alerts.js: {js_res.status_code}"
        js_code = js_res.data.decode('utf-8')

        # Check Web Audio API & Tone Profiles
        assert "AudioContext" in js_code, "Missing AudioContext reference"
        assert "sawtooth" in js_code, "Missing sawtooth oscillator for Critical"
        assert "880" in js_code and "587" in js_code, "Missing 880Hz / 587Hz tone frequencies"
        assert "440" in js_code, "Missing 440Hz sine chime for Medium"
        assert "critical: 2000" in js_code, "Missing 2s critical cooldown"
        assert "high: 8000" in js_code, "Missing 8s high cooldown"
        assert "medium: 12000" in js_code, "Missing 12s medium cooldown"
        assert "safelabsAudio" in js_code, "Missing safelabsAudio global definition"
        assert "toggleMute" in js_code, "Missing toggleMute method"
        assert "unlockAudio" in js_code, "Missing unlockAudio listener"
        print("  ✓ audio_alerts.js contains all required tone profiles, cooldowns, and autoplay unlock!")

        # Step 3: Check Header Integration in base.html
        print("\n[3] Checking Header Integration...")
        dash_res = client.get('/', follow_redirects=True)
        assert dash_res.status_code == 200
        html = dash_res.data.decode('utf-8')
        assert "audio_alerts.js" in html, "audio_alerts.js script tag missing from base template"
        assert "id=\"audioToggleBtn\"" in html, "Missing #audioToggleBtn in header"
        assert "id=\"audioUnmutedIcon\"" in html, "Missing #audioUnmutedIcon"
        assert "id=\"audioMutedIcon\"" in html, "Missing #audioMutedIcon"
        assert "pollLiveTelemetry" in html, "Missing live telemetry receiver in base template"
        print("  ✓ Header integration verified with speaker toggle button and telemetry polling receiver!")

        # Step 4: Check /api/telemetry endpoint
        print("\n[4] Checking /api/telemetry endpoint...")
        telem_res = client.get('/api/telemetry')
        assert telem_res.status_code == 200, f"Telemetry failed: {telem_res.status_code}"
        telem_data = json.loads(telem_res.data.decode('utf-8'))
        assert "highest_severity" in telem_data, "Missing highest_severity in telemetry"
        assert "unack_alerts" in telem_data, "Missing unack_alerts in telemetry"
        assert "unack_violations" in telem_data, "Missing unack_violations in telemetry"
        print(f"  ✓ /api/telemetry returned valid payload: highest_severity={telem_data.get('highest_severity')}, unack_alerts={telem_data.get('unack_alerts')}, unack_violations={telem_data.get('unack_violations')}")

        # Step 5: Check /api/camera/1/status endpoint
        print("\n[5] Checking /api/camera/1/status endpoint...")
        status_res = client.get('/api/camera/1/status')
        assert status_res.status_code == 200, f"Status failed: {status_res.status_code}"
        status_data = json.loads(status_res.data.decode('utf-8'))
        assert "severity" in status_data, "Missing severity in camera status telemetry"
        print(f"  ✓ /api/camera/1/status returned severity tier: {status_data.get('severity')}")

        print("\n🎉 ALL AUDIO ALERT ENGINE & TELEMETRY TESTS PASSED!")

if __name__ == '__main__':
    run_audio_tests()
