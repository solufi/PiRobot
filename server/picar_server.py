"""PiCar-X web controller.

Architecture:
- Background thread captures frames from Picamera2 into a shared latest_frame buffer.
- Background thread runs face tracking at ~10 Hz on the latest frame (decoupled from video).
- MJPEG /video_feed encodes & yields the latest frame at the consumer's rate.
- WebSocket /ws handles control events (drive, settings, camera, tracking, stop).
- HTTP POST endpoints kept as fallback for the legacy UI.
"""
import os, base64, functools, logging, threading, time, json, subprocess, tempfile, uuid, pathlib, math

from flask import Flask, render_template_string, request, jsonify, Response, send_file, abort
from flask_sock import Sock
from picarx import Picarx
from picamera2 import Picamera2
import cv2

from gpt_brain import GPTBrain

try:
    from robot_hat.utils import enable_speaker
except Exception:
    enable_speaker = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("picar")

app = Flask(__name__)
sock = Sock(app)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("PICAR_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))
)
px = Picarx()

AUTH_USER = os.environ.get("PICAR_USER")
AUTH_PASS = os.environ.get("PICAR_PASS")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
settings = {
    "speed": 70,
    "turn_angle": 30,
    "cam_pan": 0,
    "cam_tilt": 0,
    "tracking": False,
    "follow_me": False,
    "last_face": "OFF",
    "volume": 100,
    # Software gain boost applied on top of amixer (100 = unity, 200 = +6dB, 300 = +9.5dB)
    "volume_boost": 150,
    "cam_pan_cali": 0.0,
    "cam_tilt_cali": 0.0,
    "dir_cali": 0.0,
    # Sensors (updated by sensor_loop)
    "distance_cm": -1.0,
    "grayscale": [0, 0, 0],
    "obstacle": False,
    "cliff": False,
    # Safety: auto-stop on obstacle / cliff
    "safety": True,
    # Continuous listening with wake word
    "listening": False,
    "last_heard": "",
}

# Safety thresholds
SAFE_STOP_CM = float(os.environ.get("PICAR_SAFE_STOP_CM", "15"))
CLIFF_THRESHOLD = int(os.environ.get("PICAR_CLIFF_THRESHOLD", "200"))  # ADC value below = cliff/black
SENSOR_HZ = 10.0

# Audio config
# Use ALSA card NAMES so we are robust to renumbering across reboots.
PI_MIC_DEVICE = os.environ.get("PICAR_MIC", "plughw:CARD=Device,DEV=0")
PI_MIC_SECONDS = float(os.environ.get("PICAR_MIC_SECONDS", "5"))
TTS_DIR = pathlib.Path(tempfile.gettempdir()) / "picar_tts"
TTS_DIR.mkdir(exist_ok=True)
tts_play_lock = threading.Lock()
mic_lock = threading.Lock()  # serialize arecord usage between /voice/pi and listen_loop


def cleanup_tts_files(max_age_seconds: int = 3600):
    cutoff = time.time() - max_age_seconds
    for path in TTS_DIR.glob("tts_*.mp3"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("could not remove stale TTS file: %s", path)

# Wake word config
WAKE_WORD = os.environ.get("PICAR_WAKE_WORD", "jamal").lower()
WAKE_ALIASES = [a.strip().lower() for a in os.environ.get(
    "PICAR_WAKE_ALIASES", "djamal,jamale,gamal").split(",") if a.strip()]
LISTEN_CHUNK_SEC = float(os.environ.get("PICAR_LISTEN_CHUNK_SEC", "2.5"))
LISTEN_FOLLOWUP_SEC = float(os.environ.get("PICAR_LISTEN_FOLLOWUP_SEC", "3.5"))
# Auto-reverse threshold during follow-me: cm to keep from the user
FOLLOW_BACKUP_CM = float(os.environ.get("PICAR_FOLLOW_BACKUP_CM", "25"))
# Grace period after losing the target before switching to active search.
FOLLOW_GRACE_SEC = float(os.environ.get("PICAR_FOLLOW_GRACE_SEC", "0.8"))
# Total time to actively search before giving up (stop and wait).
FOLLOW_SEARCH_SEC = float(os.environ.get("PICAR_FOLLOW_SEARCH_SEC", "6.0"))

control = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False,
}

state_lock = threading.RLock()
motor_lock = threading.RLock()
drive_cancel = threading.Event()
task_lock = threading.Lock()
frame_lock = threading.Lock()
latest_frame = None         # numpy RGB array (raw)
latest_overlay = None       # numpy RGB array with face box drawn (when tracking on)
frame_seq = 0

CAM_SIZE = (640, 480)
JPEG_QUALITY = int(os.environ.get("PICAR_JPEG_QUALITY", "70"))
TRACKING_HZ = float(os.environ.get("PICAR_TRACKING_HZ", "20"))
# Detect at lower resolution for speed (YuNet on 320x240 is ~5ms on Pi 5).
DETECT_SIZE = (320, 240)
DETECT_SCALE_X = CAM_SIZE[0] / DETECT_SIZE[0]
DETECT_SCALE_Y = CAM_SIZE[1] / DETECT_SIZE[1]

camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": CAM_SIZE}))
camera.start()

# YuNet face detector (fast DNN, ~5ms on Pi 5 at 320x240).
YUNET_MODEL = os.environ.get(
    "PICAR_YUNET_MODEL",
    "/home/solufi/models/face_detection_yunet_2023mar.onnx",
)
face_detector = None
try:
    if os.path.exists(YUNET_MODEL):
        face_detector = cv2.FaceDetectorYN_create(
            YUNET_MODEL, "", DETECT_SIZE, 0.7, 0.3, 5000,
        )
        log.info("YuNet face detector loaded (%s)", YUNET_MODEL)
    else:
        log.warning("YuNet model not found at %s; falling back to Haar", YUNET_MODEL)
except Exception:
    log.exception("YuNet init failed; falling back to Haar")

face_cascade = None
if face_detector is None:
    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )

# Full-body person detector (HOG + linear SVM, built into OpenCV).
# Heavier than YuNet (~50ms on Pi 5 at 320x240) so we run it less often.
PERSON_DETECT_EVERY_N = int(os.environ.get("PICAR_PERSON_EVERY_N", "5"))
person_detector = None
try:
    person_detector = cv2.HOGDescriptor()
    person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    log.info("HOG person detector loaded")
except Exception:
    log.exception("HOG person detector init failed")
    person_detector = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(v, mi, ma):
    return max(mi, min(ma, v))


def update_movement():
    with motor_lock:
        angle = 0
        if control["left"]:
            angle -= settings["turn_angle"]
        if control["right"]:
            angle += settings["turn_angle"]
        px.set_dir_servo_angle(angle)

        if control["forward"] and not control["backward"]:
            px.forward(settings["speed"])
        elif control["backward"] and not control["forward"]:
            px.backward(settings["speed"])
        else:
            px.stop()


def stop_all():
    drive_cancel.set()
    with state_lock:
        for k in control:
            control[k] = False
    with motor_lock:
        px.stop()
        px.set_dir_servo_angle(0)


def safety_blocks(direction: str) -> bool:
    """Return whether safety sensors currently forbid a movement."""
    with state_lock:
        safety = settings["safety"]
        obstacle = settings["obstacle"]
        cliff = settings["cliff"]
    if not safety:
        return False
    return cliff or (obstacle and direction in {"forward", "left", "right"})


def camera_center():
    settings["cam_pan"] = 0
    settings["cam_tilt"] = 0
    px.set_cam_pan_angle(0)
    px.set_cam_tilt_angle(0)


def apply_camera():
    px.set_cam_pan_angle(settings["cam_pan"])
    px.set_cam_tilt_angle(settings["cam_tilt"])


def state_snapshot():
    with state_lock:
        return {
            "settings": dict(settings),
            "control": dict(control),
        }


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------
def capture_loop():
    global latest_frame, frame_seq
    log.info("capture thread started")
    while True:
        try:
            frame = camera.capture_array()  # RGB
            with frame_lock:
                latest_frame = frame
                frame_seq += 1
        except Exception:
            log.exception("capture error")
            time.sleep(0.1)


def detect_faces(frame_rgb):
    """Return list of (x,y,w,h,score) in CAM_SIZE coordinates.

    Uses YuNet if available (fast DNN), falls back to Haar.
    """
    if face_detector is not None:
        small = cv2.resize(frame_rgb, DETECT_SIZE)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        _, faces = face_detector.detect(bgr)
        if faces is None:
            return []
        out = []
        for f in faces:
            x = float(f[0]) * DETECT_SCALE_X
            y = float(f[1]) * DETECT_SCALE_Y
            w = float(f[2]) * DETECT_SCALE_X
            h = float(f[3]) * DETECT_SCALE_Y
            score = float(f[14]) if len(f) > 14 else 1.0
            out.append((int(x), int(y), int(w), int(h), score))
        return out
    # Haar fallback
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]


def detect_persons(frame_rgb):
    """Return list of (x,y,w,h,score) of full-body persons in CAM_SIZE coords."""
    if person_detector is None:
        return []
    small = cv2.resize(frame_rgb, DETECT_SIZE)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    try:
        rects, weights = person_detector.detectMultiScale(
            gray, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
    except Exception:
        return []
    out = []
    for (x, y, w, h), score in zip(rects, weights.flatten() if len(weights) else []):
        if score < 0.3:
            continue
        out.append((
            int(x * DETECT_SCALE_X),
            int(y * DETECT_SCALE_Y),
            int(w * DETECT_SCALE_X),
            int(h * DETECT_SCALE_Y),
            float(score),
        ))
    return out


def tracking_loop():
    global latest_overlay
    period = 1.0 / max(TRACKING_HZ, 1.0)
    cx_target, cy_target = CAM_SIZE[0] // 2, CAM_SIZE[1] // 2
    # Proportional gains: how aggressively to chase the face.
    KP_PAN = 0.05      # deg per pixel of horizontal error
    KP_TILT = 0.05
    DEAD_PX = 25       # ignore tiny errors to avoid jitter
    log.info("tracking thread started (%.1f Hz, detector=%s)",
             TRACKING_HZ, "YuNet" if face_detector else "Haar")
    last_seq = -1
    last_log_t = 0
    detect_count = 0
    # Memory of last known target for grace + search behaviours.
    # side: -1 if target was on the left of the frame, +1 if on the right.
    last_target = {"t": 0.0, "dx": 0, "w": 0, "side": 1, "kind": "face"}
    # Exponentially-smoothed target geometry (reduces detector jitter)
    smooth = {"dx": None, "dy": None, "w": None}
    EMA_ALPHA = float(os.environ.get("PICAR_TRACK_EMA", "0.35"))
    # Drive-state memory: don't re-issue same motor command every frame
    last_drive = {"action": None, "speed": 0, "turn": 0, "t": 0.0}
    DRIVE_RESEND_SEC = 0.4   # only re-send same drive after this delay
    # Larger deadband for body movement (separate from camera DEAD_PX)
    BODY_DEAD_PX = 60
    while True:
        if not settings["tracking"]:
            settings["last_face"] = "OFF"
            latest_overlay = None
            time.sleep(0.1)
            continue

        with frame_lock:
            if frame_seq == last_seq or latest_frame is None:
                frame = None
            else:
                frame = latest_frame
                last_seq = frame_seq

        if frame is None:
            time.sleep(period / 2)
            continue

        try:
            t0 = time.time()
            faces = detect_faces(frame)
            detect_ms = (time.time() - t0) * 1000.0
            detect_count += 1

            # Fallback to body detection (HOG) when no face is found AND we
            # need to follow the user. HOG is heavier so we throttle it.
            target_kind = "face"
            persons = []
            if not faces and settings["follow_me"] and person_detector is not None \
                    and (detect_count % PERSON_DETECT_EVERY_N == 0):
                persons = detect_persons(frame)
                if persons:
                    target_kind = "person"

            if time.time() - last_log_t > 5.0:
                log.info("tracker: %.1f ms/detect, %d detects", detect_ms, detect_count)
                last_log_t = time.time()
                detect_count = 0

            targets = faces if faces else persons
            now = time.time()
            if not targets:
                # Reset smoother so re-acquisition starts fresh
                smooth = {"dx": None, "dy": None, "w": None}
                # Lost target: do not stop immediately. Use grace + active search.
                if settings["follow_me"]:
                    age = now - last_target["t"]
                    side = last_target["side"] or 1
                    follow_speed = settings["speed"]
                    if age < FOLLOW_GRACE_SEC and last_target["t"] > 0:
                        # GRACE: keep moving toward where we last saw them.
                        last_dx = last_target["dx"]
                        turn = clamp(int(last_dx / 8), -30, 30)
                        px.set_dir_servo_angle(turn)
                        # If we were already close, just coast slowly forward
                        if last_target["w"] > 0 and last_target["w"] > 200:
                            px.forward(min(follow_speed, 25))
                        else:
                            px.forward(min(follow_speed, 40))
                        settings["last_face"] = f"perdu({age:.1f}s) coast"
                    elif age < FOLLOW_SEARCH_SEC:
                        # SEARCH: sweep camera and pivot the body in last seen side.
                        # Camera pan oscillation around the last side.
                        sweep = side * (35 + 20 * math.sin((now - last_target["t"]) * 4))
                        with state_lock:
                            settings["cam_pan"] = clamp(sweep, -60, 60)
                            settings["cam_tilt"] = clamp(0, -40, 40)
                            apply_camera()
                        # Body: alternate small forward/backward turns to spin in place
                        phase = int((age - FOLLOW_GRACE_SEC) / 0.7) % 2
                        if phase == 0:
                            px.set_dir_servo_angle(int(side * 30))
                            px.forward(min(follow_speed, 28))
                        else:
                            px.set_dir_servo_angle(int(-side * 30))
                            px.backward(min(follow_speed, 28))
                        settings["last_face"] = f"recherche({age:.1f}s) side={side}"
                    else:
                        px.stop()
                        settings["last_face"] = "Perdu"
                else:
                    settings["last_face"] = "Aucun visage"
                latest_overlay = None
            else:
                x, y, w, h, score = max(targets, key=lambda f: f[2] * f[3])
                cx, cy = x + w // 2, y + h // 2
                raw_dx, raw_dy = cx - cx_target, cy - cy_target

                # EMA smoothing of target geometry (reduces frame-to-frame jitter)
                if smooth["dx"] is None:
                    smooth["dx"], smooth["dy"], smooth["w"] = raw_dx, raw_dy, w
                else:
                    a = EMA_ALPHA
                    smooth["dx"] = a * raw_dx + (1 - a) * smooth["dx"]
                    smooth["dy"] = a * raw_dy + (1 - a) * smooth["dy"]
                    smooth["w"]  = a * w      + (1 - a) * smooth["w"]
                dx = int(smooth["dx"])
                dy = int(smooth["dy"])
                ws = int(smooth["w"])

                with state_lock:
                    # Proportional camera control: only move servo when error
                    # is meaningful, otherwise the servos chatter constantly.
                    if abs(dx) > DEAD_PX:
                        settings["cam_pan"] = clamp(
                            settings["cam_pan"] + KP_PAN * dx, -60, 60
                        )
                    if abs(dy) > DEAD_PX:
                        settings["cam_tilt"] = clamp(
                            settings["cam_tilt"] - KP_TILT * dy, -40, 40
                        )
                    apply_camera()
                    settings["last_face"] = f"{target_kind} dx:{dx} dy:{dy} w:{ws}"

                # Update last-known target memory
                last_target = {
                    "t": now, "dx": int(dx), "w": int(ws),
                    "side": 1 if dx >= 0 else -1, "kind": target_kind,
                }

                # ----- Drive helper with hysteresis: suppresses repeats -----
                def drive(action: str, speed: int = 0, turn: int = 0):
                    nonlocal last_drive
                    speed = int(speed); turn = int(turn)
                    same = (last_drive["action"] == action
                            and abs(last_drive["speed"] - speed) < 4
                            and abs(last_drive["turn"] - turn) < 4)
                    if same and (now - last_drive["t"]) < DRIVE_RESEND_SEC:
                        return
                    if action != last_drive["action"] or abs(last_drive["turn"] - turn) >= 4:
                        px.set_dir_servo_angle(turn)
                    if action == "forward":
                        px.forward(speed)
                    elif action == "backward":
                        px.backward(speed)
                    elif action == "stop":
                        px.stop()
                    last_drive = {"action": action, "speed": speed, "turn": turn, "t": now}

                # Follow-me drive: smart turn + ultrasonic backup
                if settings["follow_me"]:
                    if target_kind == "person":
                        far_w, near_w = 80, 280   # body wider on screen
                    else:
                        far_w, near_w = 110, 240  # face thresholds (wider deadband)
                    follow_speed = settings["speed"]
                    # Smoothed distance from sensor_loop's median filter
                    dist_now = settings.get("distance_cm", -1)
                    too_close = (isinstance(dist_now, (int, float))
                                 and 0 < dist_now < FOLLOW_BACKUP_CM)

                    edge_offset = abs(dx) > 200   # wider threshold
                    if too_close:
                        side = 1 if dx >= 0 else -1
                        drive("backward", min(follow_speed, 35), int(-side * 25))
                    elif edge_offset and ws < far_w:
                        side = 1 if dx >= 0 else -1
                        drive("backward", min(follow_speed, 35), int(-side * 30))
                    else:
                        # Proportional turn — only beyond body deadband
                        if abs(dx) <= BODY_DEAD_PX:
                            turn = 0
                        else:
                            turn = clamp(int(dx / 9), -30, 30)
                        if ws < far_w:
                            drive("forward", follow_speed, turn)
                        elif ws > near_w:
                            drive("backward", min(follow_speed, 35), turn)
                        else:
                            drive("stop", 0, turn)

                overlay = frame.copy()
                if target_kind == "person":
                    color = (255, 100, 0)  # blue/orange for body
                else:
                    color = (0, 200, 255) if settings["follow_me"] else (0, 255, 0)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                cv2.putText(overlay, f"{target_kind} {score:.2f}",
                            (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                latest_overlay = overlay
        except Exception:
            log.exception("tracking error")

        time.sleep(period)


def sensor_loop():
    """Read ultrasonic + grayscale at SENSOR_HZ; auto-stop on obstacle/cliff."""
    log.info("sensor thread started (%.1f Hz)", SENSOR_HZ)
    period = 1.0 / SENSOR_HZ
    last_warn_t = 0.0
    dist_window: list[float] = []  # rolling median filter for ultrasonic
    DIST_WIN = 5
    while True:
        try:
            raw_dist = px.get_distance()
        except Exception:
            raw_dist = -1.0
        try:
            gs = px.get_grayscale_data()
        except Exception:
            gs = [0, 0, 0]

        # Median filter: ultrasonic on PiCar can spike (-1, 400, etc.).
        if isinstance(raw_dist, (int, float)) and raw_dist > 0:
            dist_window.append(float(raw_dist))
            if len(dist_window) > DIST_WIN:
                dist_window.pop(0)
        if dist_window:
            dist = sorted(dist_window)[len(dist_window) // 2]
        else:
            dist = -1.0

        obstacle = isinstance(dist, (int, float)) and 0 < dist < SAFE_STOP_CM
        # Cliff: any sensor reads below threshold (very dark = no surface)
        cliff = any(isinstance(v, (int, float)) and v < CLIFF_THRESHOLD for v in gs)

        with state_lock:
            settings["distance_cm"] = float(dist) if isinstance(dist, (int, float)) else -1.0
            settings["grayscale"] = [int(v) if isinstance(v, (int, float)) else 0 for v in gs]
            settings["obstacle"] = obstacle
            settings["cliff"] = cliff

        # Safety guard: if moving forward and obstacle/cliff detected, hard-stop
        if settings["safety"] and (obstacle or cliff):
            forward_active = (
                control.get("forward")
                or settings["follow_me"]  # follow_me drives forward when face is far
            )
            if forward_active:
                px.stop()
                if time.time() - last_warn_t > 1.0:
                    log.warning("SAFETY stop: obstacle=%s cliff=%s dist=%.1f gs=%s",
                                obstacle, cliff, dist if isinstance(dist,(int,float)) else -1, gs)
                    last_warn_t = time.time()

        time.sleep(period)


threading.Thread(target=capture_loop, daemon=True, name="capture").start()
threading.Thread(target=tracking_loop, daemon=True, name="tracking").start()
threading.Thread(target=sensor_loop, daemon=True, name="sensor").start()
# Note: listen thread is started further below, after listen_loop is defined.


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def requires_auth(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if _request_is_authorized():
            return fn(*a, **kw)
        return Response("Auth required", 401,
                        {"WWW-Authenticate": 'Basic realm="PiCar-X"'})
    return wrapper


def _request_is_authorized():
    if not AUTH_USER:
        return True
    h = request.headers.get("Authorization", "")
    if h.startswith("Basic "):
        try:
            decoded = base64.b64decode(h[6:], validate=True).decode("utf-8")
            u, p = decoded.split(":", 1)
            return u == AUTH_USER and p == AUTH_PASS
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------
def mjpeg_gen():
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    last_seq = -1
    while True:
        with frame_lock:
            if frame_seq == last_seq:
                frame = None
            else:
                frame = (latest_overlay if (settings["tracking"] and latest_overlay is not None)
                         else latest_frame)
                if frame is not None:
                    frame = frame.copy()
                last_seq = frame_seq
        if frame is None:
            time.sleep(0.005)
            continue

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


# ---------------------------------------------------------------------------
# Control dispatcher (shared by HTTP /cmd and WebSocket)
# ---------------------------------------------------------------------------
def dispatch_action(action):
    with state_lock:
        if action in control:
            control[action] = True
        elif action.startswith("release_"):
            key = action[len("release_"):]
            if key in control:
                control[key] = False
        elif action == "stop":
            stop_all()
            return
        elif action == "center":
            control["left"] = False
            control["right"] = False
            px.set_dir_servo_angle(0)
            return
        elif action == "cam_center":
            camera_center()
            return
        else:
            return
        update_movement()


def apply_settings(speed=None, turn=None):
    with state_lock:
        if speed is not None:
            settings["speed"] = clamp(int(speed), 10, 100)
        if turn is not None:
            settings["turn_angle"] = clamp(int(turn), 5, 60)
        update_movement()


def apply_camera_settings(pan=None, tilt=None):
    with state_lock:
        if pan is not None:
            settings["cam_pan"] = clamp(int(pan), -60, 60)
        if tilt is not None:
            settings["cam_tilt"] = clamp(int(tilt), -40, 40)
        apply_camera()


# ---------------------------------------------------------------------------
# Action handlers exposed to GPT
# ---------------------------------------------------------------------------
def act_drive(direction: str, duration_ms: int = 600):
    duration_ms = max(100, min(int(duration_ms), 1500))
    if direction not in {"forward", "backward", "left", "right", "stop"}:
        return
    if direction != "stop" and safety_blocks(direction):
        stop_all()
        return
    with state_lock:
        sp = settings["speed"]
        turn_angle = settings["turn_angle"]
    if direction == "stop":
        stop_all(); return
    drive_cancel.clear()
    with motor_lock:
        turn = 0
        if direction == "left":
            turn = -turn_angle
        elif direction == "right":
            turn = turn_angle
        px.set_dir_servo_angle(turn)
        if direction == "backward":
            px.backward(sp)
        else:
            px.forward(sp)
    deadline = time.monotonic() + duration_ms / 1000.0
    while time.monotonic() < deadline and not drive_cancel.is_set():
        if safety_blocks(direction):
            drive_cancel.set()
            break
        time.sleep(0.05)
    with motor_lock:
        px.stop()
        px.set_dir_servo_angle(0)


def act_run_task(steps: list[dict]) -> dict:
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        return {"ok": False, "error": "une tâche doit contenir 1 à 8 étapes"}
    total_ms = 0
    with task_lock:
        drive_cancel.clear()
        for step in steps:
            if not isinstance(step, dict):
                return {"ok": False, "error": "étape invalide"}
            action = step.get("action")
            if action == "drive":
                direction = step.get("direction")
                duration_ms = step.get("duration_ms", 600)
                if direction not in {"forward", "backward", "left", "right", "stop"}:
                    return {"ok": False, "error": "direction invalide"}
                try:
                    duration_ms = max(100, min(int(duration_ms), 1500))
                except (TypeError, ValueError):
                    return {"ok": False, "error": "durée invalide"}
                total_ms += duration_ms
                if total_ms > 12000:
                    return {"ok": False, "error": "durée totale maximale dépassée"}
                act_drive(direction, duration_ms)
                if direction == "stop":
                    drive_cancel.clear()
            elif action == "wait":
                duration_ms = step.get("duration_ms", 100)
                try:
                    duration_ms = max(100, min(int(duration_ms), 1500))
                except (TypeError, ValueError):
                    return {"ok": False, "error": "durée invalide"}
                total_ms += duration_ms
                if total_ms > 12000:
                    return {"ok": False, "error": "durée totale maximale dépassée"}
                if drive_cancel.wait(duration_ms / 1000.0):
                    return {"ok": False, "cancelled": True}
            else:
                return {"ok": False, "error": "action invalide"}
            if drive_cancel.is_set():
                return {"ok": False, "cancelled": True}
        stop_all()
    return {"ok": True, "steps": len(steps), "duration_ms": total_ms}


def act_set_speed(speed: int):
    with state_lock:
        settings["speed"] = clamp(int(speed), 10, 100)


def act_set_camera(pan: int = None, tilt: int = None):
    apply_camera_settings(pan, tilt)


def act_set_follow_me(enabled: bool):
    with state_lock:
        settings["follow_me"] = bool(enabled)
        settings["tracking"] = bool(enabled) or settings["tracking"]
        if not enabled:
            px.stop()


def act_dance():
    for ang in (-30, 30, -30, 30, 0):
        px.set_cam_pan_angle(ang)
        time.sleep(0.18)
    with state_lock:
        settings["cam_pan"] = 0


def act_stop_all():
    with state_lock:
        settings["follow_me"] = False
        settings["tracking"] = False
    stop_all()


def act_set_volume(percent: int):
    set_volume(percent)


def act_get_distance() -> dict:
    return {"distance_cm": settings["distance_cm"], "obstacle": settings["obstacle"]}


def act_get_grayscale() -> dict:
    return {"grayscale": settings["grayscale"], "cliff": settings["cliff"]}


def act_set_safety(enabled: bool):
    with state_lock:
        settings["safety"] = bool(enabled)


def act_set_listening(enabled: bool):
    with state_lock:
        settings["listening"] = bool(enabled)
    log.info("listening %s (via tool)", settings["listening"])


# ---------------------------------------------------------------------------
# Calibration (servo zeroing — persisted by picarx into /opt/picar-x/picar-x.conf)
# ---------------------------------------------------------------------------
CALI_BOUNDS = {"cam_pan": (-20, 20), "cam_tilt": (-20, 20), "dir": (-20, 20)}


def apply_calibration(target: str, value: float):
    """target in {'cam_pan','cam_tilt','dir'}."""
    lo, hi = CALI_BOUNDS[target]
    v = float(max(lo, min(hi, float(value))))
    with state_lock:
        if target == "cam_pan":
            px.cam_pan_servo_calibrate(v)
            settings["cam_pan_cali"] = v
            px.set_cam_pan_angle(settings["cam_pan"])
        elif target == "cam_tilt":
            px.cam_tilt_servo_calibrate(v)
            settings["cam_tilt_cali"] = v
            px.set_cam_tilt_angle(settings["cam_tilt"])
        elif target == "dir":
            px.dir_servo_calibrate(v)
            settings["dir_cali"] = v
    log.info("calibration %s -> %.2f", target, v)
    return v


brain = GPTBrain({
    "drive": act_drive,
    "run_task": act_run_task,
    "set_speed": act_set_speed,
    "set_camera": act_set_camera,
    "set_follow_me": act_set_follow_me,
    "set_volume": act_set_volume,
    "dance": act_dance,
    "stop_all": act_stop_all,
    "get_distance": act_get_distance,
    "get_grayscale": act_get_grayscale,
    "set_safety": act_set_safety,
    "set_listening": act_set_listening,
})


# ---------------------------------------------------------------------------
# Audio helpers (Pi mic + TTS playback on Pi speaker)
# ---------------------------------------------------------------------------
def record_pi_mic(seconds: float, device: str = None) -> str:
    """Record from the Pi USB mic to a temp wav and return the path."""
    out = TTS_DIR / f"rec_{uuid.uuid4().hex}.wav"
    dev = device or PI_MIC_DEVICE
    cmd = ["arecord", "-D", dev, "-q", "-f", "S16_LE",
           "-r", "16000", "-c", "1", "-d", str(int(seconds)), str(out)]
    with mic_lock:
        subprocess.run(cmd, check=True, timeout=seconds + 5)
    return str(out)


# Common Whisper hallucinations on silence/noise — ignore them
WHISPER_HALLUCINATIONS = [
    "amara.org", "sous-titres", "subtitles", "merci d'avoir regard",
    "thanks for watching", "thank you for watching",
]


def _is_hallucination(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in WHISPER_HALLUCINATIONS)


def set_mic_gain_max():
    """Force USB mic capture gain to maximum (helps Whisper STT)."""
    card = os.environ.get("PICAR_MIC_CARD", "Device")
    ctl = os.environ.get("PICAR_MIC_CONTROL", "Mic")
    try:
        subprocess.run(["amixer", "-q", "-c", card, "sset", ctl, "100%", "cap"],
                       check=False, timeout=2)
        log.info("USB mic gain set to 100%% (card=%s control=%s)", card, ctl)
    except Exception:
        log.exception("set_mic_gain_max failed")


def _normalize(s: str) -> str:
    """Lowercase and strip accents/punct for wake-word matching."""
    import unicodedata, re
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _find_wake(text: str):
    """Return the (index, length) of the wake word in normalized text, or None."""
    norm = _normalize(text)
    for w in [WAKE_WORD] + WAKE_ALIASES:
        idx = norm.find(w)
        if idx >= 0:
            return idx, len(w), norm
    return None


def nod_head():
    """Quick head-nod gesture to acknowledge the wake word.

    Tilts the camera down then up then back to the user's chosen tilt.
    Non-blocking from the listen thread perspective is not strictly required,
    but we keep it short (~0.5s total).
    """
    try:
        with state_lock:
            base = settings["cam_tilt"]
        for ang in (base - 18, base + 12, base):
            ang = clamp(ang, -40, 40)
            px.set_cam_tilt_angle(ang)
            time.sleep(0.18)
    except Exception:
        log.exception("nod_head failed")


def listen_loop():
    """Background loop: record short chunks, transcribe, react to wake word."""
    log.info("listen thread started (wake='%s', aliases=%s)", WAKE_WORD, WAKE_ALIASES)
    while True:
        if not settings["listening"] or not brain.enabled:
            time.sleep(0.3)
            continue
        try:
            wav = record_pi_mic(LISTEN_CHUNK_SEC)
        except Exception:
            log.exception("listen recording failed")
            time.sleep(1.0)
            continue
        try:
            text = brain.transcribe(wav)
        except Exception:
            log.exception("listen transcribe failed")
            text = ""
        finally:
            try: os.unlink(wav)
            except Exception: pass

        if not text or _is_hallucination(text):
            log.info("listen heard but ignored (silence/halluc): %r", text[:80])
            continue

        match = _find_wake(text)
        if not match:
            log.info("listen heard (no wake): %r", text[:80])
            continue

        idx, wlen, norm = match
        # Extract command after the wake word in the same chunk
        tail = norm[idx + wlen:].strip(" ,.:;!?-")
        log.info("WAKE '%s' detected: text=%r tail=%r", WAKE_WORD, text, tail)
        with state_lock:
            settings["last_heard"] = text
        # Acknowledge with a quick head nod (non-blocking)
        threading.Thread(target=nod_head, daemon=True).start()

        cmd = tail
        if not cmd or len(cmd) < 3:
            # Wake word alone: prompt and record a follow-up
            try:
                p = tts_to_file("Oui ?")
                if p:
                    play_on_pi(p)
            except Exception:
                pass
            try:
                wav2 = record_pi_mic(LISTEN_FOLLOWUP_SEC)
                cmd = brain.transcribe(wav2)
                os.unlink(wav2)
            except Exception:
                log.exception("follow-up failed")
                continue

        cmd = (cmd or "").strip()
        if not cmd:
            continue
        log.info("processing voice command: %r", cmd)
        try:
            _process_user_message(cmd, speak_on_pi=True)
        except Exception:
            log.exception("voice command processing failed")


# Start wake-word listen thread now that listen_loop is defined.
threading.Thread(target=listen_loop, daemon=True, name="listen").start()


VOLUME_MIXER = os.environ.get("PICAR_VOLUME_MIXER", "robot-hat speaker")
VOLUME_CARD = os.environ.get("PICAR_VOLUME_CARD", "sndrpihifiberry")


def _pulse_env() -> dict:
    """Return env with PULSE_RUNTIME_PATH pointing to the solufi user socket."""
    env = os.environ.copy()
    # PulseAudio socket created by i2samp.sh for user 'solufi'
    for uid in ("1000", "1001"):
        sock_dir = f"/run/user/{uid}/pulse"
        if os.path.exists(sock_dir):
            env.setdefault("PULSE_RUNTIME_PATH", sock_dir)
            break
    return env


def play_on_pi(path: str):
    """Play an audio file through the Pi speakers (blocking, serialized).

    Routes through ALSA dmix (the 'default' device configured by i2samp.sh)
    so we don't conflict with whatever holds the hifiberry card exclusively.
    Falls back to plughw direct, then ffplay.
    """
    plughw = f"plughw:CARD={VOLUME_CARD}"
    # Configurable list of ALSA devices to try, in order
    alsa_devices = [d.strip() for d in os.environ.get(
        "PICAR_ALSA_DEVICES", f"default,{plughw}").split(",") if d.strip()]
    env = os.environ.copy()
    env["SDL_AUDIODRIVER"] = "alsa"

    # Software boost (0..3.0) — values > 1.0 amplify above 0dB.
    boost_pct = max(0, min(300, int(settings.get("volume_boost", 100))))
    boost = boost_pct / 100.0

    def _try_mpg123(dev: str) -> int | None:
        try:
            cmd = ["mpg123", "-q", "-o", "alsa", "-a", dev]
            if boost != 1.0:
                # -f factor: 32768 = unity, higher = louder (mpg123 clips)
                factor = max(0, min(98304, int(32768 * boost)))
                cmd += ["-f", str(factor)]
            cmd.append(path)
            r = subprocess.run(cmd, check=False, timeout=30, env=env)
            log.info("play_on_pi: mpg123 dev=%s rc=%d boost=%.2f",
                     dev, r.returncode, boost)
            return r.returncode
        except FileNotFoundError:
            return None
        except Exception:
            log.exception("mpg123 failed (dev=%s)", dev)
            return -1

    def _try_ffmpeg_aplay(dev: str) -> int:
        try:
            ff_cmd = ["ffmpeg", "-loglevel", "quiet", "-i", path]
            if boost != 1.0:
                # 'volume' filter accepts a multiplier; clip prevents distortion overflow
                ff_cmd += ["-filter:a", f"volume={boost:.2f}"]
            ff_cmd += ["-f", "wav", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-"]
            ff = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, env=env)
            ap = subprocess.Popen(["aplay", "-q", "-D", dev],
                                  stdin=ff.stdout, env=env)
            if ff.stdout:
                ff.stdout.close()
            rc = ap.wait(timeout=30)
            ff.wait(timeout=2)
            log.info("play_on_pi: ffmpeg|aplay dev=%s rc=%d", dev, rc)
            return rc
        except Exception:
            log.exception("ffmpeg|aplay failed (dev=%s)", dev)
            return -1

    with tts_play_lock:
        mpg123_seen = False
        for dev in alsa_devices:
            rc = _try_mpg123(dev)
            if rc is None:
                # mpg123 not installed at all — stop trying it
                if not mpg123_seen:
                    log.warning("mpg123 not installed; install with: "
                                "sudo apt-get install -y mpg123")
                mpg123_seen = True
            elif rc == 0:
                return
            # ffmpeg | aplay fallback
            if _try_ffmpeg_aplay(dev) == 0:
                return

        # Last resort: ffplay (uses default ALSA/PulseAudio)
        try:
            r = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                check=False, timeout=30, env=env,
            )
            log.info("play_on_pi: ffplay rc=%d (last resort)", r.returncode)
        except Exception:
            log.exception("ffplay failed")


def get_volume() -> int | None:
    """Return current playback volume (%) or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["amixer", "-c", VOLUME_CARD, "sget", VOLUME_MIXER],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
        # Look for first "[NN%]" occurrence
        import re
        m = re.search(r"\[(\d+)%\]", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def set_volume(percent: int) -> int:
    """Set playback volume (clamped 0..100). Returns the value applied."""
    pct = max(0, min(100, int(percent)))
    try:
        subprocess.run(
            ["amixer", "-q", "-c", VOLUME_CARD, "sset", VOLUME_MIXER, f"{pct}%"],
            check=True, timeout=2,
        )
        with state_lock:
            settings["volume"] = pct
        log.info("volume set to %d%%", pct)
    except Exception:
        log.exception("set_volume failed")
    return pct


def tts_to_file(text: str) -> str | None:
    if not brain.enabled or not text.strip():
        return None
    cleanup_tts_files()
    out = TTS_DIR / f"tts_{uuid.uuid4().hex}.mp3"
    log.info("TTS: creating file %s for text: %r", out, text[:50])
    try:
        brain.synthesize(text, str(out))
        log.info("TTS: synthesis completed, file exists: %s, size: %d", out.exists(), out.stat().st_size if out.exists() else 0)
        return str(out)
    except Exception:
        log.exception("TTS failed")
        return None


# ---------------------------------------------------------------------------
# HTML / UI
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PiCar-X</title>
<style>
html,body{min-height:100%;overflow-y:auto;}
body{background:#101010;color:#eee;font-family:system-ui,Arial;text-align:center;padding:10px;margin:0;}
h2{margin:6px 0;}
.camera-box{width:95vw;max-width:520px;margin:8px auto;background:#1c1c1c;padding:8px;border-radius:16px;}
.camera-box img{width:100%;border-radius:12px;display:block;}
.remote{display:grid;grid-template-columns:85px 85px 85px;gap:10px;justify-content:center;margin-top:10px;}
button{width:85px;height:70px;border-radius:18px;border:none;font-size:22px;background:#333;color:white;touch-action:manipulation;user-select:none;}
button:active{background:#555;}
.stop{background:#b00020;}
.small-btn{width:140px;height:45px;margin:5px;font-size:14px;}
.panel{max-width:360px;margin:10px auto;background:#1c1c1c;padding:12px;border-radius:16px;}
input[type=range]{width:100%;}
.value{color:#00e676;font-weight:bold;}
.ws{position:fixed;top:6px;right:8px;font-size:11px;color:#888;}
.ws.ok{color:#00e676;}
.ws.bad{color:#ff5252;}
.fs-btn{position:absolute;top:14px;right:14px;width:42px;height:42px;font-size:20px;border-radius:50%;background:rgba(0,0,0,0.55);}
.camera-box{position:relative;}

/* ---------- FULLSCREEN MODE ---------- */
body.fullscreen{padding:0;overflow:hidden;}
body.fullscreen h2,
body.fullscreen .panel,
body.fullscreen #faceStatus,
body.fullscreen .ws{display:none!important;}
body.fullscreen .camera-box{
  position:fixed;inset:0;width:100vw;height:100vh;max-width:none;
  margin:0;padding:0;border-radius:0;background:#000;z-index:1;
}
body.fullscreen .camera-box img{
  width:100%;height:100%;object-fit:contain;border-radius:0;
}
body.fullscreen .fs-btn{position:fixed;top:10px;right:10px;z-index:5;}
/* Top overlay: status buttons */
body.fullscreen .top-overlay{
  position:fixed;top:10px;left:10px;z-index:5;
  display:flex;flex-wrap:wrap;gap:6px;max-width:60vw;
}
body.fullscreen .top-overlay .small-btn{
  width:auto;padding:0 12px;height:38px;font-size:12px;margin:0;
  background:rgba(0,0,0,0.55);
}
/* Bottom info bar */
body.fullscreen .info-bar{
  position:fixed;bottom:8px;left:50%;transform:translateX(-50%);
  z-index:5;background:rgba(0,0,0,0.55);padding:4px 10px;border-radius:10px;
  font-size:12px;
}
/* D-pad bottom-right */
body.fullscreen .remote{
  position:fixed;bottom:14px;right:14px;z-index:5;
  margin:0;gap:6px;grid-template-columns:64px 64px 64px;
}
body.fullscreen .remote button{
  width:64px;height:54px;font-size:18px;
  background:rgba(40,40,40,0.75);border-radius:14px;
}
body.fullscreen .remote .stop{background:rgba(176,0,32,0.85);}
</style>
</head>
<body>
<div class="ws" id="wsStatus">WS: …</div>
<h2>PiCar-X</h2>

<div class="camera-box">
  <img src="/video_feed" alt="camera">
  <button class="fs-btn" id="fsBtn" title="Plein écran" onclick="toggleFullscreen()">⛶</button>
</div>

<div class="top-overlay">
  <button class="small-btn" id="trackBtn">TRACK: <span id="trackStatus">OFF</span></button>
  <button class="small-btn" id="followBtn">SUIS-MOI: <span id="followStatus">OFF</span></button>
  <button class="small-btn" id="listenBtn">👂 JAMAL: <span id="listenStatus">OFF</span></button>
  <button class="small-btn" onclick="toggleSettings()">PARAMÈTRES</button>
  <button class="small-btn" onclick="toggleCali()">🛠️ CALIBRER</button>
  <button class="small-btn" onclick="toggleChat()">🤖 CHAT</button>
</div>
<p>Détection: <span class="value" id="faceStatus">---</span></p>
<p class="info-bar" style="font-size:13px;">
  📏 <span id="distVal" class="value">--</span> cm
  &nbsp;·&nbsp; 🌗 <span id="gsVal" class="value">--</span>
  &nbsp;·&nbsp; <span id="safetyBadge" style="cursor:pointer;padding:2px 8px;border-radius:8px;background:#1b5e20;font-size:11px;">SAFETY ON</span>
</p>

<div class="remote">
  <div></div><button data-cmd="forward">▲</button><div></div>
  <button data-cmd="left">◀</button>
  <button class="stop" data-once="stop">STOP</button>
  <button data-cmd="right">▶</button>
  <div></div><button data-cmd="backward">▼</button><div></div>
</div>

<div class="panel" id="settingsPanel" style="display:none;">
  <h3>Réglages</h3>
  <label>Vitesse: <span class="value" id="speedValue">40</span></label>
  <input id="speed" type="range" min="10" max="100" value="40">
  <label>Angle virage: <span class="value" id="turnValue">30</span></label>
  <input id="turn" type="range" min="5" max="60" value="30">
  <label>Pan caméra: <span class="value" id="panValue">0</span></label>
  <input id="pan" type="range" min="-60" max="60" value="0">
  <label>Tilt caméra: <span class="value" id="tiltValue">0</span></label>
  <input id="tilt" type="range" min="-40" max="40" value="0">
  <label>🔊 Volume: <span class="value" id="volumeValue">100</span>%</label>
  <input id="volume" type="range" min="0" max="100" value="100">
  <label>🔊 Boost (gain logiciel): <span class="value" id="boostValue">150</span>%</label>
  <input id="boost" type="range" min="50" max="300" value="150">
  <br><br>
  <button class="small-btn" data-once="center">CENTER</button>
  <button class="small-btn" data-once="cam_center">CAM CENTER</button>
</div>

<div class="panel" id="caliPanel" style="display:none;">
  <h3>🛠️ Calibration zéro</h3>
  <p style="font-size:12px;color:#aaa;margin:4px 0;">Ajuste pour que la caméra et les roues pointent droit. Sauvegardé automatiquement.</p>
  <label>Caméra Tilt (haut/bas): <span class="value" id="caliTiltValue">0</span></label>
  <input id="caliTilt" type="range" min="-20" max="20" step="0.5" value="0">
  <label>Caméra Pan (gauche/droite): <span class="value" id="caliPanValue">0</span></label>
  <input id="caliPan" type="range" min="-20" max="20" step="0.5" value="0">
  <label>Roues directrices: <span class="value" id="caliDirValue">0</span></label>
  <input id="caliDir" type="range" min="-20" max="20" step="0.5" value="0">
  <br><br>
  <button class="small-btn" onclick="resetCali()">RESET (0,0,0)</button>
</div>

<div class="panel" id="chatPanel" style="display:none;">
  <h3>🤖 Parle au robot</h3>
  <div id="chatLog" style="text-align:left;max-height:180px;overflow-y:auto;background:#0a0a0a;padding:8px;border-radius:8px;font-size:13px;margin-bottom:8px;"></div>
  <input id="chatInput" type="text" placeholder="Tape un message..." style="width:100%;padding:8px;border-radius:8px;border:none;background:#222;color:#eee;font-size:14px;">
  <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;justify-content:center;">
    <button class="small-btn" id="chatSend">Envoyer</button>
    <button class="small-btn" id="micPi" style="background:#0277bd;">🎤 Micro robot (5s)</button>
    <button class="small-btn" id="micBrowser" style="background:#6a1b9a;">🎙️ Micro tél.</button>
  </div>
  <label style="display:block;margin-top:6px;font-size:12px;"><input type="checkbox" id="speakOnPi" checked> Réponse parlée par le robot</label>
  <audio id="ttsPlayer" style="display:none;"></audio>
</div>

<script>
const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
let ws = null, wsReady = false, reconnectDelay = 500;
const wsBadge = document.getElementById("wsStatus");

function connect(){
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { wsReady = true; reconnectDelay = 500; wsBadge.textContent = "WS ✓"; wsBadge.className="ws ok"; };
  ws.onclose = () => { wsReady = false; wsBadge.textContent = "WS …"; wsBadge.className="ws bad"; setTimeout(connect, reconnectDelay); reconnectDelay = Math.min(reconnectDelay*2, 5000); };
  ws.onerror = () => { try { ws.close(); } catch(e){} };
  ws.onmessage = ev => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") applyState(msg.data);
    } catch(e){}
  };
}
function wsSend(obj){
  if (wsReady) { ws.send(JSON.stringify(obj)); return true; }
  // Fallback HTTP
  if (obj.type === "cmd") {
    fetch("/cmd", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:"action="+obj.action});
  }
  return false;
}
let volumeUserDragging = false;
function applyState(s){
  if (!s) return;
  if (s.settings) {
    document.getElementById("faceStatus").innerText = s.settings.last_face || "---";
    document.getElementById("trackStatus").innerText = s.settings.tracking ? "ON" : "OFF";
    document.getElementById("followStatus").innerText = s.settings.follow_me ? "ON" : "OFF";
    document.getElementById("listenStatus").innerText = s.settings.listening ? "ON" : "OFF";
    document.getElementById("listenBtn").style.background = s.settings.listening ? "#1b5e20" : "#333";
    if (!volumeUserDragging && typeof s.settings.volume === "number") {
      document.getElementById("volume").value = s.settings.volume;
      document.getElementById("volumeValue").innerText = s.settings.volume;
    }
    if (typeof s.settings.volume_boost === "number") {
      const b = document.getElementById("boost");
      if (b && b !== document.activeElement) {
        b.value = s.settings.volume_boost;
        document.getElementById("boostValue").innerText = s.settings.volume_boost;
      }
    }
    if (typeof s.settings.distance_cm === "number") {
      const d = s.settings.distance_cm;
      const el = document.getElementById("distVal");
      el.innerText = (d > 0 ? d.toFixed(1) : "--");
      el.style.color = s.settings.obstacle ? "#ff5252" : "#00e676";
    }
    if (Array.isArray(s.settings.grayscale)) {
      const el = document.getElementById("gsVal");
      el.innerText = s.settings.grayscale.join(",");
      el.style.color = s.settings.cliff ? "#ff5252" : "#00e676";
    }
    if (typeof s.settings.safety === "boolean") {
      const b = document.getElementById("safetyBadge");
      b.innerText = s.settings.safety ? "SAFETY ON" : "SAFETY OFF";
      b.style.background = s.settings.safety ? "#1b5e20" : "#b00020";
    }
  }
}

function toggleSettings(){
  const p = document.getElementById("settingsPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
}
function toggleFullscreen(){
  const isFs = document.body.classList.toggle("fullscreen");
  const btn = document.getElementById("fsBtn");
  if (btn) btn.innerText = isFs ? "✕" : "⛶";
  // Best-effort native fullscreen (mobile / desktop)
  try {
    if (isFs && document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(()=>{});
    } else if (!isFs && document.fullscreenElement && document.exitFullscreen) {
      document.exitFullscreen().catch(()=>{});
    }
  } catch (e) {}
}
// Sync class if user exits native fullscreen via ESC
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && document.body.classList.contains("fullscreen")) {
    document.body.classList.remove("fullscreen");
    const btn = document.getElementById("fsBtn");
    if (btn) btn.innerText = "⛶";
  }
});
function toggleChat(){
  const p = document.getElementById("chatPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
}
function toggleCali(){
  const p = document.getElementById("caliPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
  if (p.style.display === "block") loadCali();
}

async function loadCali(){
  try {
    const r = await fetch("/calibration");
    const d = await r.json();
    document.getElementById("caliPan").value = d.cam_pan;
    document.getElementById("caliPanValue").innerText = d.cam_pan;
    document.getElementById("caliTilt").value = d.cam_tilt;
    document.getElementById("caliTiltValue").innerText = d.cam_tilt;
    document.getElementById("caliDir").value = d.dir;
    document.getElementById("caliDirValue").innerText = d.dir;
  } catch(e) {}
}
function bindCali(id, target){
  const el = document.getElementById(id);
  const label = document.getElementById(id+"Value");
  let t = null;
  el.addEventListener("input", () => {
    label.innerText = el.value;
    clearTimeout(t);
    t = setTimeout(() => {
      wsSend({type:"calibration", target: target, value: parseFloat(el.value)});
    }, 60);
  });
}
bindCali("caliPan", "cam_pan");
bindCali("caliTilt", "cam_tilt");
bindCali("caliDir", "dir");
function resetCali(){
  for (const [id,target] of [["caliPan","cam_pan"],["caliTilt","cam_tilt"],["caliDir","dir"]]) {
    document.getElementById(id).value = 0;
    document.getElementById(id+"Value").innerText = "0";
    wsSend({type:"calibration", target: target, value: 0});
  }
}

document.getElementById("trackBtn").addEventListener("click", () => {
  const on = document.getElementById("trackStatus").innerText !== "ON";
  wsSend({type:"tracking", value: on});
});
document.getElementById("followBtn").addEventListener("click", () => {
  const on = document.getElementById("followStatus").innerText !== "ON";
  wsSend({type:"follow_me", value: on});
});
document.getElementById("safetyBadge").addEventListener("click", () => {
  const on = document.getElementById("safetyBadge").innerText.indexOf("ON") < 0;
  wsSend({type:"safety", value: on});
});
document.getElementById("listenBtn").addEventListener("click", () => {
  const on = document.getElementById("listenStatus").innerText !== "ON";
  wsSend({type:"listening", value: on});
});

// ----- Chat / Voice -----
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const ttsPlayer = document.getElementById("ttsPlayer");
function appendChat(role, text){
  const div = document.createElement("div");
  div.style.margin = "4px 0";
  div.innerHTML = "<b style='color:" + (role==="user"?"#00e676":"#64b5f6") + "'>" + role + ":</b> " + text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}
async function sendChat(message){
  if (!message) return;
  appendChat("user", message);
  const speakOnPi = document.getElementById("speakOnPi").checked;
  try {
    const r = await fetch("/chat", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({message: message, speak_on_pi: speakOnPi})});
    const data = await r.json();
    if (!data.ok) { appendChat("robot", "⚠️ " + (data.error||"erreur")); return; }
    appendChat("robot", data.reply);
    if (data.tts && !speakOnPi) { ttsPlayer.src = data.tts; ttsPlayer.play().catch(()=>{}); }
  } catch(e) { appendChat("robot", "⚠️ " + e.message); }
}
document.getElementById("chatSend").addEventListener("click", () => {
  const msg = chatInput.value.trim(); chatInput.value = ""; sendChat(msg);
});
chatInput.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); document.getElementById("chatSend").click(); }});

// Pi mic: triggers server-side recording
document.getElementById("micPi").addEventListener("click", async () => {
  const btn = document.getElementById("micPi");
  const orig = btn.innerText; btn.innerText = "🎤 ÉCOUTE..."; btn.disabled = true;
  try {
    const r = await fetch("/voice/pi", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({seconds: 5})});
    const data = await r.json();
    if (data.transcript) appendChat("user 🎤", data.transcript);
    if (data.reply) appendChat("robot", data.reply);
    if (!data.ok) appendChat("robot", "⚠️ " + (data.error||""));
  } catch(e) { appendChat("robot", "⚠️ " + e.message); }
  btn.innerText = orig; btn.disabled = false;
});

// Browser mic (requires HTTPS or localhost for getUserMedia)
let mediaRec = null, recChunks = [];
document.getElementById("micBrowser").addEventListener("click", async () => {
  const btn = document.getElementById("micBrowser");
  if (mediaRec && mediaRec.state === "recording") {
    mediaRec.stop(); return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    appendChat("robot", "⚠️ Micro navigateur indisponible (HTTPS requis)."); return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    mediaRec = new MediaRecorder(stream);
    recChunks = [];
    mediaRec.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
    mediaRec.onstop = async () => {
      btn.innerText = "🎙️ Micro tél."; btn.style.background = "#6a1b9a";
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(recChunks, {type: mediaRec.mimeType || "audio/webm"});
      const fd = new FormData();
      fd.append("audio", blob, "rec.webm");
      fd.append("speak_on_pi", document.getElementById("speakOnPi").checked ? "true" : "false");
      const r = await fetch("/voice/browser", {method:"POST", body: fd});
      const data = await r.json();
      if (data.transcript) appendChat("user 🎙️", data.transcript);
      if (data.reply) appendChat("robot", data.reply);
      if (data.tts && !document.getElementById("speakOnPi").checked) { ttsPlayer.src = data.tts; ttsPlayer.play().catch(()=>{}); }
    };
    mediaRec.start();
    btn.innerText = "⏹️ Stop"; btn.style.background = "#b00020";
  } catch(e) { appendChat("robot", "⚠️ " + e.message); }
});

function bindRange(id, key){
  const el = document.getElementById(id);
  const label = document.getElementById(id+"Value");
  el.addEventListener("input", () => {
    label.innerText = el.value;
    const msg = {type:"settings"};
    msg[key] = parseInt(el.value, 10);
    wsSend(msg);
  });
}
bindRange("speed","speed");
bindRange("turn","turn");
bindRange("pan","pan");
bindRange("tilt","tilt");

// Volume slider (separate WS message type, debounced)
(function(){
  const el = document.getElementById("volume");
  const label = document.getElementById("volumeValue");
  let t = null;
  el.addEventListener("input", () => {
    label.innerText = el.value;
    volumeUserDragging = true;
    clearTimeout(t);
    t = setTimeout(() => {
      wsSend({type:"volume", value: parseInt(el.value, 10)});
      setTimeout(() => { volumeUserDragging = false; }, 300);
    }, 80);
  });
})();
// Boost slider (software gain >100%)
(function(){
  const el = document.getElementById("boost");
  const label = document.getElementById("boostValue");
  if (!el) return;
  let t = null;
  el.addEventListener("input", () => {
    label.innerText = el.value;
    clearTimeout(t);
    t = setTimeout(() => {
      wsSend({type:"volume_boost", value: parseInt(el.value, 10)});
    }, 120);
  });
})();

function press(action){ wsSend({type:"cmd", action: action}); }
function release(action){ wsSend({type:"cmd", action: "release_" + action}); }
function once(action){ wsSend({type:"cmd", action: action}); }

document.querySelectorAll("button[data-cmd]").forEach(b => {
  const action = b.dataset.cmd;
  const onDown = e => { e.preventDefault(); press(action); };
  const onUp   = e => { e.preventDefault(); release(action); };
  b.addEventListener("touchstart", onDown, {passive:false});
  b.addEventListener("touchend", onUp, {passive:false});
  b.addEventListener("touchcancel", onUp, {passive:false});
  b.addEventListener("mousedown", onDown);
  b.addEventListener("mouseup", onUp);
  b.addEventListener("mouseleave", onUp);
});
document.querySelectorAll("button[data-once]").forEach(b => {
  b.addEventListener("click", () => once(b.dataset.once));
});

// Keyboard
const keyMap = {ArrowUp:"forward", ArrowDown:"backward", ArrowLeft:"left", ArrowRight:"right",
                w:"forward", s:"backward", a:"left", d:"right"};
const held = {};
window.addEventListener("keydown", e => {
  const k = keyMap[e.key]; if (!k || held[k]) return;
  held[k] = true; press(k);
});
window.addEventListener("keyup", e => {
  const k = keyMap[e.key]; if (!k) return;
  held[k] = false; release(k);
  if (e.key === " ") once("stop");
});

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/")
@requires_auth
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
@requires_auth
def video():
    return Response(mjpeg_gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
@requires_auth
def status():
    return jsonify(state_snapshot()["settings"])


@app.route("/tracking", methods=["POST"])
@requires_auth
def tracking_route():
    with state_lock:
        settings["tracking"] = request.form.get("tracking") == "true"
    return "ok"


@app.route("/settings", methods=["POST"])
@requires_auth
def set_settings_route():
    apply_settings(request.form.get("speed"), request.form.get("turn"))
    return jsonify(settings)


@app.route("/camera_settings", methods=["POST"])
@requires_auth
def camera_settings_route():
    apply_camera_settings(request.form.get("pan"), request.form.get("tilt"))
    return jsonify(settings)


@app.route("/cmd", methods=["POST"])
@requires_auth
def cmd_route():
    action = request.form.get("action") or ""
    dispatch_action(action)
    return "ok"


@app.route("/volume", methods=["GET", "POST"])
@requires_auth
def volume_route():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "volume" in data:
            set_volume(data["volume"])
        elif request.form.get("volume") is not None:
            set_volume(request.form.get("volume"))
    return jsonify({"volume": get_volume()})


@app.route("/say", methods=["POST"])
@requires_auth
def say_route():
    """Quick TTS test: synthesize text and play it on the Pi speaker."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "Bonjour, je suis le robot.").strip()[:500]
    if not brain.enabled:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY non configurée"}), 503
    path = tts_to_file(text)
    if not path:
        return jsonify({"ok": False, "error": "TTS failed"}), 500
    threading.Thread(target=play_on_pi, args=(path,), daemon=True).start()
    return jsonify({"ok": True, "text": text, "tts": f"/tts/{pathlib.Path(path).name}"})


@app.route("/calibration", methods=["GET", "POST"])
@requires_auth
def calibration_route():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        for k in ("cam_pan", "cam_tilt", "dir"):
            if k in data:
                apply_calibration(k, data[k])
    with state_lock:
        return jsonify({
            "cam_pan": settings["cam_pan_cali"],
            "cam_tilt": settings["cam_tilt_cali"],
            "dir": settings["dir_cali"],
        })


# ---------------------------------------------------------------------------
# ChatGPT / Voice routes
# ---------------------------------------------------------------------------
def _process_user_message(message: str, speak_on_pi: bool) -> dict:
    """Send a user message to the brain, optionally play TTS on the Pi."""
    if not brain.enabled:
        return {"ok": False, "error": "OPENAI_API_KEY non configurée"}
    result = brain.chat(message[:2000])
    reply = result.get("reply", "")
    tts_path = tts_to_file(reply)
    tts_id = pathlib.Path(tts_path).name if tts_path else None
    if speak_on_pi and tts_path:
        threading.Thread(target=play_on_pi, args=(tts_path,), daemon=True).start()
    return {
        "ok": True,
        "transcript": message,
        "reply": reply,
        "actions": result.get("actions", []),
        "tts": f"/tts/{tts_id}" if tts_id else None,
    }


@app.route("/chat", methods=["POST"])
@requires_auth
def chat_route():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    speak = bool(data.get("speak_on_pi", False))
    if not message:
        return jsonify({"ok": False, "error": "message vide"}), 400
    return jsonify(_process_user_message(message, speak))


@app.route("/voice/pi", methods=["POST"])
@requires_auth
def voice_pi_route():
    """Record from the Pi USB mic, transcribe with Whisper, run brain, play TTS on Pi."""
    if not brain.enabled:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY non configurée"}), 503
    data = request.get_json(silent=True) or {}
    try:
        seconds = float(data.get("seconds", PI_MIC_SECONDS))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "durée invalide"}), 400
    seconds = max(1.0, min(seconds, 15.0))
    try:
        wav = record_pi_mic(seconds)
    except subprocess.CalledProcessError as e:
        return jsonify({"ok": False, "error": f"arecord failed: {e}"}), 500
    try:
        text = brain.transcribe(wav)
    finally:
        try: os.unlink(wav)
        except Exception: pass
    if not text:
        return jsonify({"ok": True, "transcript": "", "reply": "Je n'ai rien entendu.", "actions": []})
    return jsonify(_process_user_message(text, speak_on_pi=True))


@app.route("/voice/browser", methods=["POST"])
@requires_auth
def voice_browser_route():
    """Receive an audio blob from the browser, transcribe, run brain."""
    if not brain.enabled:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY non configurée"}), 503
    f = request.files.get("audio")
    if not f:
        return jsonify({"ok": False, "error": "audio manquant"}), 400
    speak = request.form.get("speak_on_pi", "false").lower() == "true"
    suffix = "." + (f.filename.rsplit(".", 1)[-1] if "." in (f.filename or "") else "webm")
    tmp = TTS_DIR / f"in_{uuid.uuid4().hex}{suffix}"
    f.save(str(tmp))
    try:
        text = brain.transcribe(str(tmp))
    finally:
        try: os.unlink(tmp)
        except Exception: pass
    if not text:
        return jsonify({"ok": True, "transcript": "", "reply": "Je n'ai rien compris.", "actions": []})
    return jsonify(_process_user_message(text, speak_on_pi=speak))


@app.route("/tts/<name>")
@requires_auth
def tts_route(name):
    # Prevent path traversal
    if "/" in name or "\\" in name or not (name.startswith("tts_") and name.endswith(".mp3")):
        abort(404)
    p = TTS_DIR / name
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="audio/mpeg")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@sock.route("/ws")
def ws_endpoint(ws):
    if not _request_is_authorized():
        ws.close()
        return
    log.info("ws client connected")
    # initial state push
    try:
        ws.send(json.dumps({"type": "state", "data": state_snapshot()}))
    except Exception:
        return

    # background pusher: send state every 1s
    stop_evt = threading.Event()

    def pusher():
        while not stop_evt.is_set():
            try:
                ws.send(json.dumps({"type": "state", "data": state_snapshot()}))
            except Exception:
                stop_evt.set()
                return
            stop_evt.wait(0.4)

    t = threading.Thread(target=pusher, daemon=True)
    t.start()

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "cmd":
                dispatch_action(msg.get("action", ""))
            elif mtype == "settings":
                apply_settings(msg.get("speed"), msg.get("turn"))
            elif mtype == "camera":
                apply_camera_settings(msg.get("pan"), msg.get("tilt"))
            elif mtype == "tracking":
                with state_lock:
                    settings["tracking"] = bool(msg.get("value"))
            elif mtype == "follow_me":
                act_set_follow_me(bool(msg.get("value")))
            elif mtype == "safety":
                act_set_safety(bool(msg.get("value")))
            elif mtype == "listening":
                with state_lock:
                    settings["listening"] = bool(msg.get("value"))
                log.info("listening %s", settings["listening"])
            elif mtype == "volume":
                set_volume(msg.get("value", 100))
            elif mtype == "volume_boost":
                v = max(0, min(300, int(msg.get("value", 100))))
                with state_lock:
                    settings["volume_boost"] = v
                log.info("volume_boost set to %d%%", v)
            elif mtype == "calibration":
                target = msg.get("target")
                if target in ("cam_pan", "cam_tilt", "dir") and "value" in msg:
                    apply_calibration(target, msg["value"])
            elif mtype == "ping":
                try: ws.send(json.dumps({"type": "pong", "t": msg.get("t")}))
                except Exception: break
    finally:
        stop_evt.set()
        log.info("ws client disconnected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        stop_all()
        camera_center()
        # Power on the robot-hat speaker amplifier (GPIO gate)
        if enable_speaker:
            try:
                enable_speaker()
                log.info("robot-hat speaker amplifier enabled")
            except Exception:
                log.exception("enable_speaker failed")
        # Ensure USB mic capture gain is at max (USB mics often default to 0%)
        set_mic_gain_max()
        v = get_volume()
        if v is not None:
            settings["volume"] = v
            log.info("initial volume: %d%%", v)
        # Sync calibration values from picarx
        settings["cam_pan_cali"] = float(getattr(px, "cam_pan_cali_val", 0.0))
        settings["cam_tilt_cali"] = float(getattr(px, "cam_tilt_cali_val", 0.0))
        settings["dir_cali"] = float(getattr(px, "dir_cali_val", 0.0))
        log.info("calibration: pan=%.1f tilt=%.1f dir=%.1f",
                 settings["cam_pan_cali"], settings["cam_tilt_cali"], settings["dir_cali"])
        # Auto-enable wake-word listening if GPT brain is ready
        if brain.enabled and os.environ.get("PICAR_AUTOLISTEN", "1") == "1":
            with state_lock:
                settings["listening"] = True
            log.info("wake-word listening AUTO-ENABLED (wake='%s', aliases=%s)",
                     WAKE_WORD, WAKE_ALIASES)

        # Startup chime: announce that Jamal is ready. Cached on disk so
        # subsequent boots play instantly without an API call.
        def _startup_announce():
            try:
                msg = os.environ.get(
                    "PICAR_STARTUP_MESSAGE",
                    "Jamal est prêt ! Dis Jamal pour me parler.",
                )
                cache_dir = pathlib.Path(os.path.expanduser("~/.cache/picar"))
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache = cache_dir / "startup.mp3"
                if not cache.exists() and brain.enabled:
                    log.info("generating startup chime TTS...")
                    brain.synthesize(msg, str(cache))
                if cache.exists():
                    log.info("playing startup chime")
                    play_on_pi(str(cache))
                else:
                    # Fallback: short beep via ffmpeg sine wave
                    subprocess.run(
                        ["ffmpeg", "-loglevel", "quiet", "-f", "lavfi",
                         "-i", "sine=frequency=880:duration=0.25",
                         "-f", "wav", "-"],
                        check=False, timeout=3,
                        stdout=subprocess.PIPE,
                    )
            except Exception:
                log.exception("startup announce failed")
        threading.Thread(target=_startup_announce, daemon=True, name="startup").start()

        port = int(os.environ.get("PICAR_PORT", 5000))
        log.info("PiCar-X server starting on 0.0.0.0:%d (auth=%s)", port, bool(AUTH_USER))
        app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
    finally:
        stop_all()
        camera.stop()
