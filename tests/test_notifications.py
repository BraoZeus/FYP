import os
import sys
import time
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from notifications import dispatch_hazard_alert, _ALERT_COOLDOWNS, _COOLDOWN_LOCK

def test_telegram_connectivity():
    from dotenv import load_dotenv
    import requests
    load_dotenv()
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip("\"'")
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip("\"'")

    assert token, "TELEGRAM_BOT_TOKEN is missing"
    assert chat_id, "TELEGRAM_CHAT_ID is missing"

    url = f"https://api.telegram.org/bot{token}/getMe"
    resp = requests.get(url, timeout=6)
    assert resp.status_code == 200, f"Telegram getMe failed with status {resp.status_code}"
    data = resp.json()
    assert data.get("ok") is True, f"Telegram getMe error: {data}"
    print(f"[PASS] Telegram connected: {data['result']['username']}")

def test_dispatch_logic():
    # Verify cooldown dictionary mechanics
    with _COOLDOWN_LOCK:
        _ALERT_COOLDOWNS.clear()

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # First dispatch triggers successfully
    dispatch_hazard_alert("fire", location="Lab A", frame_bgr=dummy_frame)
    cooldown_key = ("fire", "lab a")
    with _COOLDOWN_LOCK:
        assert cooldown_key in _ALERT_COOLDOWNS, "Cooldown key should be registered"
        first_time = _ALERT_COOLDOWNS[cooldown_key]

    # Immediate second dispatch is suppressed by cooldown
    dispatch_hazard_alert("fire", location="Lab A", frame_bgr=dummy_frame)
    with _COOLDOWN_LOCK:
        assert _ALERT_COOLDOWNS[cooldown_key] == first_time, "Duplicate alert was not suppressed by cooldown"

    # Different lab or hazard type should not be blocked
    dispatch_hazard_alert("fire", location="Lab B", frame_bgr=dummy_frame)
    with _COOLDOWN_LOCK:
        assert ("fire", "lab b") in _ALERT_COOLDOWNS, "Separate lab should have independent cooldown"

    print("[PASS] Alert dispatch cooldown logic passed.")

if __name__ == "__main__":
    test_telegram_connectivity()
    test_dispatch_logic()
    print("All notification unit tests passed!")
