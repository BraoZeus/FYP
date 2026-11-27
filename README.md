# SAFELABS

**An Automated Computer Vision Pipeline for Safety Compliance and Prohibited Item Detection in Academic Laboratories**

SAFELABS is an edge-deployed, web-based laboratory safety monitoring platform built with Flask, OpenCV, PyTorch, and Ultralytics YOLOv8. It continuously analyzes closed-circuit television (CCTV) RTSP feeds in real time to verify mandatory personal protective equipment (PPE), detect prohibited items on chemical workbenches, and trigger immediate fire and smoke emergency alerts while enforcing privacy-by-design with encrypted evidence storage and role-based access control.

---

## Key Features

### 1. Real-Time 10-Class Computer Vision Pipeline
- **Low-Latency RTSP Ingestion (`vision.py`)**: Non-blocking producer-consumer capture thread using OpenCV/FFmpeg (`rtsp_transport;tcp`, `fflags;nobuffer`, `flags;low_delay`), automatic reconnection watchdog, RTSP credential masking in logs/UI, and ~25 FPS MJPEG browser streaming (`/video_feed`).
- **Mandatory PPE Compliance (`safety_engine.py`)**: Person tracking via YOLOv8n + ByteTrack (`safelabs_tracker.yaml`) paired with anatomical body-zone association (`head_zone`, `torso_zone`, `hands_zone`) to verify **Lab Coat**, **Protective Gloves**, and **Face Mask**. Multiple missing PPE items on the same person are consolidated into a single violation record.
- **Benchtop Contraband Gating (`safety_engine.py`)**: Dedicated YOLOv8n contraband predictor gated by a resolution-scaled workbench polygon (`BENCH_ROI`) via `cv2.pointPolygonTest` on bottom-center bounding-box anchors `(x_center, y_max)` to detect **Smartphone**, **Bottle**, **Laptop**, **Backpack**, and food/drink items strictly on active bench surfaces.
- **Fire & Smoke Combustion Detection (`hazard_engine.py`)**: Dedicated `fire_smoke.pt` sub-engine with overexposed light-source luminance filtering (`>= 238` / `>= 240` pixel saturation ratio and ceiling fixture guard) and an asymmetric 10-sample rolling queue (`>= 2/10` for Fire, `>= 5/10` for Smoke).

### 2. Stateful Temporal Debouncing & Incident Lifecycle
- **3.0-Second Grace Period (`violation_tracker.py`)**: Wall-clock temporal debouncing (`GRACE_PERIOD_SECONDS = 3.0`) filters out brief occlusions and single-frame flickers before logging PPE or contraband infractions.
- **30-Second Montage Compilation**: Continuous violations lasting `>= 30s` automatically stitch the initial (`3s`) and persistent (`30s`) frames into a single side-by-side evidence montage and update the existing database record in place (zero duplicate spam rows).
- **Timetable Rule Binding (`database.py`, `routes/classes.py`)**: Links live detection rules (`MonitoringConfig`) to active `ClassSchedule` sessions (day of week, time window, and semester dates) with General Lab fallback when no class is scheduled.

### 3. Emergency Alerting & Hardware Controls
- **ISO 8201 / NFPA 72 Temporal-3 Siren (`static/js/audio_alerts.js`)**: Zero-dependency Web Audio API evacuation siren (`180s` latch, `880 Hz -> 587.33 Hz` T3 cycle) triggered strictly on Fire or Smoke emergencies (PPE/contraband remains visual-only), with a navbar Sound Control dropdown (**Reset Sound**, **Test Siren**, **Mute All**).
- **Telegram Alert Dispatch (`notifications.py`)**: Non-blocking emergency notifications delivering an annotated evidence snapshot (`sendPhoto`) plus a synthesized spoken voice alert (`gTTS` -> `sendVoice`) with a 5-minute per-hazard cooldown.
- **Live GPU / CPU Switching & Objective 2 Benchmarking (`safety_engine.py`, `templates/monitoring.html`)**: Switch live inference between CUDA GPU and host CPU with zero restarts and leak-free predictor/VRAM cleanup, alongside a dynamic Objective 2 model evaluation and benchmark comparison table in **Detection Rules**.

### 4. Security & Privacy-by-Design (PDPA 2010 / GDPR Aligned)
- **Zero Plaintext Disk Storage (`crypto_utils.py`, `routes/evidence.py`)**: Live frames are processed exclusively in volatile RAM. Confirmed violation snapshots are hashed with **SHA-256** and encrypted in RAM using **AES-256-GCM** (`12`-byte CSPRNG nonce + authentication tag) before saving to `static/evidence/*.jpg.enc`.
- **In-Memory Decryption & Integrity Verification**: Evidence images are decrypted strictly in RAM (`BytesIO`), verified against their stored SHA-256 checksum (blocking tampered files with `HTTP 403` and logging `evidence_integrity_failure`), and protected against path traversal (`os.path.basename`).
- **Role-Based Access Control (`auth_utils.py`)**: Server-side `@role_required` enforcement returning `HTTP 403` across three roles:
  - `admin` — Full system, user management, database audit logs, live feeds, violations, schedules, and hardware controls.
  - `afm` — Admin & Facilities Management: live feeds, violations/hazards triage, detection rules, and read-only class schedule.
  - `academic_coordinator` — Class schedule CRUD and read-only detection rules (strictly blocked from CCTV feeds, evidence, and admin routes).
- **Authentication & Hardening (`app.py`, `routes/auth.py`, `database.py`)**: `PBKDF2-HMAC-SHA256` password hashing (`16`-byte random salt), strong password complexity policy, mandatory temporary password rotation, IP rate limiting (`5` attempts / `60s`), `15`-minute account lockout after `5` failed logins, CSRF protection, security headers (`CSP`, `X-Frame-Options`, `X-Content-Type-Options`, production `HSTS`), SQLite `WAL` mode, immutable `AuditLog`, and automated **30-day rolling evidence & log retention purge**.

---

## Performance Summary (Objective 2 Evaluation)

| Metric | Target | Achieved |
| :--- | :--- | :--- |
| **Overall 10-Class Accuracy** | `>= 85.0%` mAP@50 | **`87.4%` mAP@50** (`93.2%` Lab Coat, `88.5%` Gloves, `86.4%` Face Mask, `85.8%` Contraband, `86.9%` Fire/Smoke) |
| **Edge Inference Latency (GPU)** | `<= 50.0 ms` (`>= 20 FPS`) | **`34.2 ms`** mean (`29.2 FPS` on NVIDIA RTX 3050) |
| **Combustion Recall (Fire / Smoke)** | `< 5.0%` False Negatives | **`100.0%` Recall** (`0.0%` FN, `0` glare false positives) |
| **Temporal Debounce Gate** | `3.0 s` sustained confirmation | **`3.0 s`** wall-clock latch + `30 s` montage compilation |
| **Evidence Storage Security** | Zero plaintext exposure | **AES-256-GCM + SHA-256** (`.jpg.enc` only, 30-day auto-purge) |

---

## Project Structure

```text
SAFELABS/
├── app.py                     # Flask app factory, security headers, SQLite WAL, DB seeding & 30-day retention daemon
├── database.py                # SQLAlchemy ORM models (User, CameraFeed, ClassSchedule, MonitoringConfig, ViolationLog, EmergencyAlert, AuditLog)
├── safety_engine.py           # YOLOv8 PPE/contraband inference, IoU & body-zone logic, GPU/CPU controller, Objective 2 catalog & benchmarker
├── hazard_engine.py           # Fire & smoke detection, luminance glare rejection, asymmetric rolling queues
├── violation_tracker.py       # 3.0s temporal debounce engine, spatial track handoff, single per-person PPE consolidation & 30s montage builder
├── vision.py                  # Threaded RTSP/USB ingestion, stall watchdog, MJPEG streaming & credential masking
├── crypto_utils.py            # AES-256-GCM authenticated encryption & SHA-256 integrity verification
├── auth_utils.py              # RBAC constants, @role_required decorator & password complexity validation
├── audit_utils.py             # Centralized security event logging to AuditLog
├── notifications.py           # Non-blocking Telegram photo + gTTS voice alert dispatcher
├── time_utils.py              # Malaysian Time (UTC+8) and dd/mm/yy formatting utilities
├── safelabs_tracker.yaml      # Tuned ByteTrack tracking parameters
├── routes/                    # Flask Blueprints (main, auth, admin, classes, violations, alerts, video, evidence, api)
├── static/
│   ├── js/audio_alerts.js     # Web Audio API ISO 8201 / NFPA 72 Temporal-3 Fire & Smoke siren
│   └── evidence/              # Encrypted evidence vault (*.jpg.enc only)
├── templates/                 # Jinja2 + Tailwind CSS role-aware views
├── models/                    # Model weights (best(4).pt, best(1..3).pt, fire_smoke.pt, yolov8n.pt)
└── tests/                     # Automated unit, integration, RBAC, security, audio & system evaluation suites
```

---

## Prerequisites

- **Python**: 3.10+ (3.12 recommended)
- **Hardware**: NVIDIA CUDA-capable GPU recommended for real-time 25–30 FPS inference (CPU fallback supported natively)
- **Camera**: RTSP IP camera (e.g., Hikvision) or local USB webcam

---

## Installation & Setup

### 1. Clone the Repository & Create a Virtual Environment
```bash
git clone <repository-url>
cd FYP
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install psutil requests
```
> For NVIDIA GPU acceleration, install the CUDA-enabled PyTorch build matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Generate Encryption Key & Configure `.env`
Generate a secure 256-bit AES evidence key:
```bash
python -c "from crypto_utils import generate_evidence_key; print(generate_evidence_key())"
```

Create a `.env` file in the project root:
```env
# Camera RTSP Configuration (or set to 0 for local webcam)
SAFELABS_RTSP_URL=rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
SAFELABS_CAMERA_1_SOURCE=rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
SAFELABS_FFMPEG_OPTIONS=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|buffer_size;1024000|stimeout;10000000

# Security & Cryptography (REQUIRED)
SAFELABS_EVIDENCE_KEY=<paste-generated-base64-aes256-key>
SAFELABS_SECRET_KEY=<generate-64-char-random-hex-secret>
SAFELABS_API_KEY=<edge-worker-api-key>

# Model & Runtime Settings
SAFELABS_DETECTION_BACKEND=hybrid
SAFELABS_COMPUTE_DEVICE=cuda
SAFELABS_MODEL_PATH=models/best(4).pt
SAFELABS_PERSON_MODEL_PATH=models/yolov8n.pt
SAFELABS_RETENTION_DAYS=30
SAFELABS_ENABLE_AUTO_PURGE=1

# Optional: Telegram Emergency Dispatch
TELEGRAM_BOT_TOKEN=<your-telegram-bot-token>
TELEGRAM_CHAT_ID=<your-telegram-chat-id>
```

### 4. Run the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser. On first launch, SAFELABS initializes the SQLite database (`instance/safelabs.db`), runs schema migrations, seeds default accounts/cameras/hazard rules, and starts the 30-day rolling retention purge daemon.

---

## Default Seeded Accounts

| Role | Default Username | Default Password | Forced Change on First Login |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `admin` | No |
| **Admin & Facilities Management (AFM)** | `afm` | `afm` | Yes |
| **Academic Coordinator** | `academic_coordinator` | `academic_coordinator` | Yes |

> Change all default credentials immediately before deploying in a shared or production environment.

---

## Running Tests & Verification

```bash
# Role-Based Access Control (RBAC) & Template Guards
pytest tests/test_rbac_access.py -v

# Temporal Debounce, Spatial Handoff & Single Per-Person PPE Consolidation
python tests/test_violation_tracker.py

# Benchtop Contraband ROI & Confidence Gates
python tests/test_benchtop_contraband.py

# Class Schedule Semester Dates & MonitoringConfig Rules
python tests/test_class_semester_dates_and_ppe.py

# Cryptography (AES-256-GCM & SHA-256) & Security Suite
python tests/test_security_cryptography.py

# Web Audio Fire & Smoke Evacuation Siren Verification
python tests/test_audio_alerts.py

# Full 4-Domain System Evaluation Suite
python tests/run_full_system_evaluation.py
```

---

## Documentation
- [System Architecture](SYSTEM_ARCHITECTURE.md)
- [Memory Leak Prevention & Resource Management](MEMORY_LEAK_PREVENTION.md)
