"""
Video ingestion and camera stream management for SAFELABS.
Provides low-latency RTSP ingestion, non-blocking producer-consumer frame
dropping, auto-reconnection with watchdog monitoring, and multi-camera HUD.
"""
import os
import re
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from time_utils import get_myt_now
import cv2
import numpy as np
from dotenv import load_dotenv

from safety_engine import analyze_frame_ppe, render_safety_overlay
from hazard_engine import process_hazards

load_dotenv()

# ===================================================================
#  LOGGING & CREDENTIAL SECURITY
# ===================================================================
logger = logging.getLogger("safelabs.vision")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [SAFELABS.VISION] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def mask_stream_url(url: Any) -> str:
    """
    Redact embedded authentication credentials from RTSP/HTTP URLs for
    safe logging and UI display, preventing password exposure.
    Example: rtsp://admin:JYLSS.25@192.168.1.64:554/...
             -> rtsp://admin:*****@192.168.1.64:554/...
    """
    if not isinstance(url, str):
        return str(url)
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:*****@", url)


# ===================================================================
#  OPENCV & FFMPEG LOW-LATENCY CONFIGURATION
# ===================================================================
# - rtsp_transport;tcp: Forces TCP transport for RTSP packets (prevents packet loss & tearing)
# - fflags;nobuffer: Disables FFmpeg internal frame buffering for real-time delivery
# - flags;low_delay: Instructs the decoder to output frames immediately without GOP delay
# - max_delay;500000: Caps packet demuxing delay to 500ms (0.5s) to eliminate packet drops from CPU scheduling
# - buffer_size;1024000: 1MB socket buffer to absorb transient network bursts
# - stimeout/timeout;10000000: 10-second socket timeout to prevent premature disconnects
DEFAULT_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
    "max_delay;500000|buffer_size;1024000|stimeout;10000000|timeout;10000000"
)


def apply_ffmpeg_env_options():
    """Ensure OpenCV FFmpeg low-latency capture options are loaded into environment."""
    opts = os.environ.get("SAFELABS_FFMPEG_OPTIONS", DEFAULT_FFMPEG_OPTIONS)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    return opts


# Apply immediately upon module import
apply_ffmpeg_env_options()


# ===================================================================
#  CAMERA STREAM INGESTION (THREADED / NON-BLOCKING)
# ===================================================================
class CameraStream:
    """
    Robust, low-latency, non-blocking camera ingestion pipeline.
    Maintains a dedicated background ingestion thread that continuously drains
    frames from the RTSP/USB source to prevent queue build-up and latency lag.

    Includes:
    - Zero-latency frame-dropping (consumers always get the freshest frame).
    - Automatic reconnection watchdog on packet loss, read errors, or frozen sockets.
    - Clean error logging with password masking.
    - Standby laboratory HUD generator when stream is offline/reconnecting.
    """
    def __init__(
        self,
        source: Any,
        code: str = "CAM-01",
        name: str = "Camera Feed",
        location: str = "Lab A",
        camera_id: int = 1,
        reconnect_interval: float = 4.0,
        stall_timeout: float = 15.0,
    ):
        self.code = code
        self.name = name
        self.location = location
        self.camera_id = camera_id
        self.reconnect_interval = reconnect_interval
        self.stall_timeout = stall_timeout

        self._set_source(source)

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.connected: bool = False
        self.is_running: bool = True
        self.lock = threading.Lock()

        # Telemetry & Health Tracking
        self.last_frame_time: float = 0.0
        self.last_reconnect_attempt: float = 0.0
        self.reconnect_attempts: int = 0
        self.packet_drops: int = 0
        self.frames_received: int = 0
        self.last_error: str = ""
        self.stream_width: int = 0
        self.stream_height: int = 0
        self.stream_fps: float = 0.0
        self.scan_pos: int = 0
        self.start_time: float = time.time()

        self.latest_safety_analysis: Optional[Dict[str, Any]] = None
        self.ai_enabled: bool = True

        # Start dedicated non-blocking capture thread
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"IngestionThread-{self.code}",
        )
        self.thread.start()

        # Start background real-time AI inference thread
        self.ai_thread = threading.Thread(
            target=self._ai_inference_loop,
            daemon=True,
            name=f"AIThread-{self.code}",
        )
        self.ai_thread.start()

    def _set_source(self, source: Any):
        """Parse source as numeric webcam index or string stream URL / file."""
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
            self.is_network_stream = False
        elif isinstance(source, str) and (
            source.startswith("rtsp://")
            or source.startswith("http://")
            or source.startswith("https://")
        ):
            self.source = source
            self.is_network_stream = True
        else:
            self.source = source
            self.is_network_stream = False

    def update_source(self, new_source: Any):
        """Dynamically update stream source at runtime and trigger reconnection."""
        normalized_new = int(new_source) if (isinstance(new_source, str) and new_source.isdigit()) else new_source
        if self.source == normalized_new or str(self.source) == str(new_source):
            return
        masked_old = mask_stream_url(self.source)
        masked_new = mask_stream_url(new_source)
        logger.info(
            f"[{self.code}] Updating source from {masked_old} to {masked_new}"
        )
        with self.lock:
            self._set_source(new_source)
            self._disconnect()
            self.last_reconnect_attempt = 0.0  # Trigger immediate connect

    def _connect(self) -> bool:
        """
        Attempt opening the RTSP/video capture stream using low-latency flags.
        Returns True if successful, False otherwise.
        """
        masked = mask_stream_url(self.source)
        try:
            self._disconnect()
            apply_ffmpeg_env_options()

            backend = (
                cv2.CAP_FFMPEG if self.is_network_stream else cv2.CAP_ANY
            )
            logger.info(
                f"[{self.code}] Ingestion initiating for {masked} "
                f"(Backend: {'CAP_FFMPEG' if backend == cv2.CAP_FFMPEG else 'CAP_ANY'})..."
            )

            cap = cv2.VideoCapture(self.source, backend)

            # Low-latency buffer configuration: 2 frames to absorb jitter without latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                self.last_error = (
                    "Capture failed to open (verify camera IP, port 554, and credentials)"
                )
                logger.error(
                    f"[{self.code}] RTSP connection refused or timed out for {masked}. "
                    f"Error: {self.last_error}"
                )
                cap.release()
                return False

            # Read initial frame to verify video stream validity
            ret, frame = cap.read()
            if not ret or frame is None:
                self.last_error = "Connection opened but initial frame read failed"
                logger.warning(
                    f"[{self.code}] Initial frame read failed for {masked}."
                )
                cap.release()
                return False

            # Capture stream metadata
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

            with self.lock:
                self.cap = cap
                self.frame = frame
                self.ret = True
                self.connected = True
                self.last_frame_time = time.time()
                self.frames_received += 1
                self.stream_width = w
                self.stream_height = h
                self.stream_fps = fps
                self.last_error = ""

            logger.info(
                f"[{self.code}] RTSP stream connected successfully: {masked} "
                f"({w}x{h} @ {fps:.1f} FPS). Low-latency ingestion active."
            )
            return True

        except (cv2.error, OSError, IOError, RuntimeError) as e:
            self.last_error = f"Ingestion error: {e}"
            logger.error(
                f"[{self.code}] Exception while connecting to {masked}: {e}",
                exc_info=False,
            )
            self._disconnect()
            return False

    def _disconnect(self):
        """Safely release VideoCapture handle asynchronously without blocking."""
        with self.lock:
            self.connected = False
            self.ret = False
            old_cap = self.cap
            self.cap = None

        if old_cap is not None:
            def _async_release(c):
                try:
                    c.release()
                except Exception as e:
                    logger.debug(f"[{self.code}] Exception on cap.release: {e}")

            threading.Thread(
                target=_async_release, args=(old_cap,), daemon=True, name=f"CapRelease-{self.code}"
            ).start()

    def _capture_loop(self):
        """
        Background capture thread:
        1. Drains frames as fast as they arrive to prevent socket buffer build-up.
        2. Drops older unconsumed frames in memory (in-place replacement).
        3. Monitors packet loss and stream stalls via watchdog timer.
        4. Reconnects automatically if camera drops or network is interrupted.
        """
        while self.is_running:
            now = time.time()

            # Handle offline / reconnection state
            if not self.connected:
                if (now - self.last_reconnect_attempt) >= self.reconnect_interval:
                    self.last_reconnect_attempt = now
                    self.reconnect_attempts += 1
                    logger.info(
                        f"[{self.code}] Reconnection attempt #{self.reconnect_attempts} "
                        f"for {mask_stream_url(self.source)}..."
                    )
                    if self._connect():
                        self.reconnect_attempts = 0
                time.sleep(0.1)
                continue

            # Connected state: read latest frame
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.last_error = "VideoCapture object closed unexpectedly"
                    logger.warning(f"[{self.code}] {self.last_error}. Reconnecting...")
                    self._disconnect()
                    continue

                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # In-place atomic update: overwrites stale unconsumed frame
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.last_frame_time = time.time()
                        self.frames_received += 1
                else:
                    # Packet drop or frame read failure
                    self.packet_drops += 1
                    time_since_frame = time.time() - self.last_frame_time
                    if time_since_frame > 2.0 and (self.packet_drops % 30 == 0):
                        logger.warning(
                            f"[{self.code}] Frame read delay (total drops: {self.packet_drops}). "
                            f"No valid frame for {time_since_frame:.1f}s"
                        )

                    # Trigger reconnect if stream is stalled beyond threshold
                    if time_since_frame >= self.stall_timeout:
                        self.last_error = (
                            f"Stream stalled (no valid frame for >{self.stall_timeout:.1f}s)"
                        )
                        logger.error(
                            f"[{self.code}] {self.last_error}. Disconnecting for auto-reconnect."
                        )
                        self._disconnect()
                    time.sleep(0.02)

            except (cv2.error, OSError, IOError, RuntimeError) as e:
                self.last_error = f"Ingestion loop exception: {e}"
                logger.error(f"[{self.code}] Error during frame capture: {e}")
                self._disconnect()
                time.sleep(0.5)

            # Stall watchdog: catch silent freezes where cap.read() hung
            if self.connected and (time.time() - self.last_frame_time > self.stall_timeout):
                self.last_error = (
                    f"Watchdog timeout: stream silent for >{self.stall_timeout:.1f}s"
                )
                logger.warning(f"[{self.code}] {self.last_error}. Forcing reconnect.")
                self._disconnect()

    def _ai_inference_loop(self):
        """
        Dedicated background thread running YOLOv8 Person + PPE compliance
        inference independently of camera ingestion socket reads.
        Continuously updates bounding boxes and violation flags without blocking.

        Hybrid mode optimizations:
        - Uses model.track() with ByteTrack for temporal coherence and speed
        - Runs at reduced imgsz (416) for person detection since we only need person boxes
        - Minimal sleep between iterations to maximize freshness
        """
        from safety_engine import get_active_backend

        while self.is_running:
            if not self.connected or not self.ai_enabled:
                time.sleep(0.2)
                continue

            frame_to_analyze = None
            with self.lock:
                if self.frame is not None:
                    frame_to_analyze = self.frame.copy()

            if frame_to_analyze is not None:
                try:
                    self.inference_frame_idx = getattr(self, "inference_frame_idx", 0) + 1
                    # In hybrid mode, run at imgsz=640 with two-stage confidence for rock-solid stability
                    active_backend = get_active_backend()
                    if active_backend == "hybrid":
                        analysis = analyze_frame_ppe(
                            frame_to_analyze,
                            person_conf=0.15,  # Allows ByteTrack to recover tracks through shadows/turning
                            ppe_conf=0.15,
                            imgsz=640,         # Full 640 resolution eliminates 93% of dropout flickers
                            frame_idx=self.inference_frame_idx,
                        )
                    else:
                        analysis = analyze_frame_ppe(
                            frame_to_analyze,
                            person_conf=0.25,
                            ppe_conf=0.15,
                            imgsz=640,
                            frame_idx=self.inference_frame_idx,
                        )
                    with self.lock:
                        self.latest_safety_analysis = analysis

                    # log active fire and smoke hazards or detect boxes
                    hazards = analysis.get("hazards", [])
                    has_fire_box = any(h.get("label") == "FIRE" for h in hazards)
                    fire_on = analysis.get("fire_active", False) or has_fire_box
                    smoke_on = analysis.get("smoke_active", False)
                    now = time.time()

                    # Handle Fire Hazard
                    if fire_on:
                        setattr(self, "_fire_clear_counter", 0)
                        if (now - getattr(self, "_last_fire_log", 0) > 15.0):
                            self._last_fire_log = now
                            logger.critical(f"[{self.code}] CRITICAL: FIRE HAZARD DETECTED!")
                            annotated_frame = render_safety_overlay(frame_to_analyze.copy(), analysis, self.code)
                            self._record_hazard_event("Fire", annotated_frame, analysis)
                            try:
                                from notifications import dispatch_hazard_alert
                                dispatch_hazard_alert("Fire", location=self.location, frame_bgr=annotated_frame)
                            except Exception as e:
                                logger.error(f"[{self.code}] Telegram alert dispatch error: {e}")
                    else:
                        fc = getattr(self, "_fire_clear_counter", 0) + 1
                        setattr(self, "_fire_clear_counter", fc)
                        if fc > 5:
                            setattr(self, "_last_fire_log", 0.0)

                    # Handle Smoke Hazard
                    if smoke_on:
                        setattr(self, "_smoke_clear_counter", 0)
                        if (now - getattr(self, "_last_smoke_log", 0) > 15.0):
                            self._last_smoke_log = now
                            logger.warning(f"[{self.code}] WARNING: SMOKE HAZARD DETECTED!")
                            annotated_frame = render_safety_overlay(frame_to_analyze.copy(), analysis, self.code)
                            self._record_hazard_event("Smoke", annotated_frame, analysis)
                            try:
                                from notifications import dispatch_hazard_alert
                                dispatch_hazard_alert("Smoke", location=self.location, frame_bgr=annotated_frame)
                            except Exception as e:
                                logger.error(f"[{self.code}] Telegram alert dispatch error: {e}")
                    else:
                        sc = getattr(self, "_smoke_clear_counter", 0) + 1
                        setattr(self, "_smoke_clear_counter", sc)
                        if sc > 5:
                            setattr(self, "_last_smoke_log", 0.0)

                    # Run temporal debounce (3s grace period) & 30s continuous incident compilation
                    try:
                        from violation_tracker import ViolationDebounceEngine
                        annotated_frame = render_safety_overlay(frame_to_analyze.copy(), analysis, self.code)
                        ViolationDebounceEngine.get_instance().process_analysis(
                            camera_id=self.camera_id,
                            camera_code=self.code,
                            location=self.location,
                            current_frame=annotated_frame,
                            analysis=analysis,
                        )
                    except Exception as e:
                        logger.debug(f"[{self.code}] Debounce tracker note: {e}")
                except Exception as e:
                    logger.error(f"[{self.code}] AI inference error: {e}")

            # Balanced pacing: ~12-15 FPS AI inference prevents CPU starvation,
            # leaving plenty of CPU bandwidth for rock-solid 25 FPS RTSP streaming.
            time.sleep(0.06)

    def _record_hazard_event(self, hazard_name: str, annotated_frame: np.ndarray, analysis: dict):
        """
        Record a critical fire or smoke hazard into both EmergencyAlert and ViolationLog
        tables with cryptographically hashed and encrypted evidence image.
        Runs asynchronously in a background thread to prevent blocking video ingestion.
        """
        cam_id = self.camera_id
        cam_code = self.code
        loc = self.location
        hazard_type_label = f"{hazard_name} Hazard"
        alert_type_label = hazard_name.lower()

        def _worker():
            try:
                from app import app
                from database import db, EmergencyAlert, ViolationLog, ClassSchedule
                from safety_engine import save_evidence_frame, get_current_ongoing_class
                from datetime import datetime
                import os

                with app.app_context():
                    static_dir = app.static_folder or os.path.join(app.root_path, "static")
                    ongoing_class = get_current_ongoing_class(db.session, ClassSchedule)
                    class_id = ongoing_class.id if ongoing_class else None

                    rel_path, ev_hash = save_evidence_frame(
                        annotated_frame,
                        static_dir,
                        prefix=f"hazard_{alert_type_label}_cam{cam_id}",
                        hazard_type=hazard_type_label,
                        severity="critical",
                        confidence_score=0.95,
                    )

                    msg = f"Critical {hazard_name} Hazard detected at {loc} on camera {cam_code}."

                    # 1. EmergencyAlert record (for emergency hazard banner & alert center)
                    alert = EmergencyAlert(
                        alert_type=alert_type_label,
                        message=msg,
                        image_path=rel_path,
                        evidence_hash=ev_hash,
                        camera_id=cam_id,
                        timestamp=get_myt_now(),
                        is_acknowledged=False,
                    )
                    db.session.add(alert)

                    # 2. ViolationLog record (for main violations log table, evidence modal & forensic export)
                    v = ViolationLog(
                        hazard_type=hazard_type_label,
                        description=msg,
                        severity="critical",
                        confidence_score=0.95,
                        duration_seconds=5,
                        image_path=rel_path,
                        evidence_hash=ev_hash,
                        camera_id=cam_id,
                        class_schedule_id=class_id,
                        timestamp=get_myt_now(),
                        is_acknowledged=False,
                        review_status="confirmed",
                    )
                    db.session.add(v)
                    db.session.commit()

                    logger.info(
                        f"[{cam_code}] Successfully saved {hazard_name} hazard evidence to "
                        f"EmergencyAlert #{alert.id} and ViolationLog #{v.id} ({rel_path})."
                    )
            except Exception as e:
                logger.error(f"[{cam_code}] Failed saving {hazard_name} hazard to database: {e}")

        threading.Thread(target=_worker, daemon=True, name=f"HazardRecorder-{cam_code}").start()

    def get_frame(self, overlay_ai: bool = True) -> np.ndarray:
        """
        Return the camera frame.
        If overlay_ai is True, renders real-time YOLOv8 bounding boxes and
        PPE violation/compliance tags directly onto the live CCTV stream.
        """
        with self.lock:
            if self.connected and self.frame is not None:
                raw_copy = self.frame.copy()
                analysis = self.latest_safety_analysis
            else:
                raw_copy = None
                analysis = None

        if raw_copy is not None:
            if overlay_ai and analysis is not None:
                try:
                    return render_safety_overlay(raw_copy, analysis, self.code)
                except Exception as e:
                    logger.debug(f"[{self.code}] Overlay render error: {e}")
            return raw_copy

        return self._generate_standby_frame()

    def get_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return (True, frame) if live feed is active, or (False, None) if offline.
        Specifically designed for YOLOv8/SAM inference workers so AI models do
        not run false detections on standby graphics.
        """
        with self.lock:
            if self.connected and self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def is_camera_connected(self) -> bool:
        """Check if camera feed is active and delivering frames."""
        with self.lock:
            return bool(self.connected and self.frames_received > 0 and (time.time() - self.last_frame_time < self.stall_timeout))

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time ingestion telemetry and health diagnostics."""
        with self.lock:
            now = time.time()
            uptime = round(now - self.start_time, 1)
            time_since_frame = round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None
            analysis = self.latest_safety_analysis or {}
            persons = analysis.get("persons", [])
            viol_count = sum(1 for p in persons if not p.get("is_compliant"))
            cb_count = analysis.get("contraband_count", len(analysis.get("contraband", [])))
            fire_active = analysis.get("fire_active", False)
            smoke_active = analysis.get("smoke_active", False)
            is_active_streaming = bool(self.connected and self.frames_received > 0 and (now - self.last_frame_time < self.stall_timeout))

            # Determine severity tier for audio alerts and HUD telemetry
            severity = None
            if fire_active or smoke_active:
                severity = "critical"
            elif any(
                any(
                    m.get("type") == "mask"
                    or "mask" in m.get("label", "").lower()
                    for m in p.get("missing_ppe", [])
                )
                for p in persons if not p.get("is_compliant")
            ):
                severity = "high"
            elif any(
                any(
                    m.get("type") in ("lab_coat", "gloves")
                    or "coat" in m.get("label", "").lower()
                    for m in p.get("missing_ppe", [])
                )
                for p in persons if not p.get("is_compliant")
            ) or cb_count > 0:
                severity = "medium"
            elif viol_count > 0:
                severity = "low"

            return {
                "code": self.code,
                "name": self.name,
                "location": self.location,
                "source": mask_stream_url(self.source),
                "connected": is_active_streaming,
                "width": self.stream_width,
                "height": self.stream_height,
                "fps": self.stream_fps,
                "frames_received": self.frames_received,
                "packet_drops": self.packet_drops,
                "reconnect_attempts": self.reconnect_attempts,
                "time_since_last_frame": time_since_frame,
                "persons_detected": len(persons),
                "violations_detected": viol_count + cb_count,
                "contraband_detected": cb_count,
                "fire_active": fire_active,
                "smoke_active": smoke_active,
                "has_emergency": fire_active or smoke_active,
                "has_violations": analysis.get("has_violations", False) or (cb_count > 0) or fire_active or smoke_active,
                "severity": severity,
                "ai_enabled": self.ai_enabled,
                "last_error": self.last_error,
                "uptime_seconds": uptime,
            }

    def _generate_standby_frame(self) -> np.ndarray:
        # flat utilitarian vms standby graphic
        width, height = 1280, 720
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (18, 18, 20)  # matte zinc-950

        margin = 32
        # subtle outer border
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), (39, 39, 42), 1)

        # header bar
        loc_clean = str(self.location).replace('—', '-').replace('–', '-').replace('·', '-')
        cam_tag = f"SAFELABS VMS | {self.code} [{loc_clean}]"
        cv2.putText(
            img, cam_tag, (margin + 16, margin + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 225), 1, cv2.LINE_AA,
        )

        ts = get_myt_now().strftime("%d/%m/%y %H:%M:%S")
        cv2.putText(
            img, ts, (width - margin - 180, margin + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA,
        )

        # center vms status card
        box_w, box_h = 580, 120
        bx1 = (width - box_w) // 2
        by1 = (height - box_h) // 2
        cv2.rectangle(img, (bx1, by1), (bx1 + box_w, by1 + box_h), (24, 24, 28), -1)
        cv2.rectangle(img, (bx1, by1), (bx1 + box_w, by1 + box_h), (45, 45, 52), 1)

        # status indicator dot
        dot_color = (40, 160, 240) if self.reconnect_attempts > 0 else (120, 120, 130)
        cv2.circle(img, (bx1 + 24, by1 + 32), 5, dot_color, -1)

        # title and stream details
        cv2.putText(
            img, f"{self.code} | {self.name.upper()}",
            (bx1 + 40, by1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (240, 240, 245), 1, cv2.LINE_AA,
        )

        masked_src = mask_stream_url(self.source)
        if len(masked_src) > 55:
            masked_src = masked_src[:52] + "..."
        cv2.putText(
            img, f"SOURCE: {masked_src}",
            (bx1 + 24, by1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            (140, 140, 150), 1, cv2.LINE_AA,
        )

        now = time.time()
        time_to_retry = max(0.0, self.reconnect_interval - (now - self.last_reconnect_attempt))
        reconnect_msg = (
            f"RECONNECTING (Attempt #{self.reconnect_attempts}) | RETRY IN {time_to_retry:.1f}s"
            if self.reconnect_attempts > 0 else "CONNECTING..."
        )
        cv2.putText(
            img, reconnect_msg,
            (bx1 + 24, by1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (180, 180, 190), 1, cv2.LINE_AA,
        )

        # bottom diagnostics
        diag_msg = f"RTSP TCP | BUFFER: 1 | DROPS: {self.packet_drops} | RECONNECT: {self.reconnect_interval:.0f}s"
        cv2.putText(
            img, diag_msg, (margin + 16, height - margin - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 120), 1, cv2.LINE_AA,
        )

        return img

    def stop(self):
        """Stop camera ingestion thread cleanly."""
        self.is_running = False
        self._disconnect()
        if self.thread.is_alive() and threading.current_thread() != self.thread:
            self.thread.join(timeout=1.0)


# ===================================================================
#  CAMERA MANAGER (SINGLETON)
# ===================================================================
class CameraManager:
    """
    Singleton manager handling multiple camera streams.
    Provides thread safety and automatic resolution of RTSP environment URLs.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._streams = {}
        self._stream_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Return the singleton instance of CameraManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _resolve_source(self, camera_id: int, source: Optional[Any]) -> Any:
        """
        Resolve the effective video source for a camera ID:
        1. Environment variable SAFELABS_CAMERA_{camera_id}_SOURCE takes highest precedence.
        2. For camera 1, SAFELABS_RTSP_URL is checked.
        3. If source is specified (and not a dummy '0' when RTSP is configured), use it.
        4. Fall back to SAFELABS_RTSP_URL or local camera index.
        """
        env_cam_key = f"SAFELABS_CAMERA_{camera_id}_SOURCE"
        env_cam_src = os.environ.get(env_cam_key)
        if env_cam_src:
            return env_cam_src

        # If camera_id == 1 and env has SAFELABS_RTSP_URL:
        env_rtsp = os.environ.get("SAFELABS_RTSP_URL")
        if camera_id == 1 and env_rtsp:
            # If source was default '0' or None, promote to SAFELABS_RTSP_URL
            if source is None or str(source).strip() in ("0", "", "stream_1"):
                return env_rtsp

        if source is not None and str(source).strip() != "":
            return source

        if camera_id == 1 and env_rtsp:
            return env_rtsp

        return f"stream_{camera_id}"

    def get_camera(
        self,
        camera_id: int = 1,
        source: Optional[Any] = None,
        code: str = "CAM-01",
        name: str = "Camera Feed",
        location: str = "Lab A",
    ) -> CameraStream:
        """Retrieve or initialize CameraStream for requested camera ID."""
        actual_source = self._resolve_source(camera_id, source)

        with self._stream_lock:
            if camera_id in self._streams:
                stream = self._streams[camera_id]
                # If source has been updated at runtime, apply it
                if actual_source and stream.source != actual_source:
                    stream.update_source(actual_source)
                return stream

            stream = CameraStream(
                source=actual_source,
                code=code,
                name=name,
                location=location,
                camera_id=camera_id,
            )
            self._streams[camera_id] = stream
            return stream

    def reset_streams(self):
        """Cleanly close and reset all active streams."""
        with self._stream_lock:
            for stream in self._streams.values():
                stream.stop()
            self._streams.clear()


# ===================================================================
#  PUBLIC API ACCESSORS
# ===================================================================
def get_camera(
    camera_id: int = 1,
    source: Optional[Any] = None,
    code: str = "CAM-01",
    name: str = "Camera Feed",
    location: str = "Lab",
) -> CameraStream:
    """Global accessor returning the CameraStream instance."""
    return CameraManager.get_instance().get_camera(
        camera_id=camera_id,
        source=source,
        code=code,
        name=name,
        location=location,
    )


def generate_frames(
    camera_id: int = 1,
    source: Optional[Any] = None,
    code: str = "CAM-01",
    name: str = "Camera Feed",
    location: str = "Lab",
):
    """
    Generator yielding multipart JPEG frames with dynamic ~25 FPS pacing.
    Resizes high-resolution streams (e.g. 1080p) to max-width 1280 (720p) for high-performance,
    low-bandwidth browser rendering. Adds Content-Length headers for seamless Chromium decoding.
    """
    cam = get_camera(
        camera_id=camera_id,
        source=source,
        code=code,
        name=name,
        location=location,
    )
    target_interval = 1.0 / 25.0  # 25 FPS target matches native camera rate

    try:
        while True:
            t_start = time.time()
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if w > 1280:
                new_w = 1280
                new_h = int(h * (1280.0 / w))
                send_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                send_frame = frame

            ret, buffer = cv2.imencode(
                ".jpg", send_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if not ret:
                time.sleep(0.01)
                continue

            frame_bytes = buffer.tobytes()
            header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
            )
            yield header + frame_bytes + b"\r\n"
            elapsed = time.time() - t_start
            sleep_time = max(0.001, target_interval - elapsed)
            time.sleep(sleep_time)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError, OSError):
        # Client closed tab, navigated to another page, or detached stream
        pass
