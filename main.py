#!/usr/bin/env python3
"""
=============================================================================
  Droid Dustbin Robot — Arm Raise Detection Module
  File   : main.py
  Target : Raspberry Pi 4 + Raspberry Pi Camera Module (CSI / Picamera2)
  OS     : Raspberry Pi OS Bookworm (64-bit)
  Python : 3.11
=============================================================================

  CAMERA PIPELINE — COMPLETE DESIGN RATIONALE
  ════════════════════════════════════════════

  Why XRGB8888 is the mandatory format on Bookworm
  ──────────────────────────────────────────────────
  libcamera (the kernel-level camera stack on Bookworm) processes sensor
  data through its ISP and outputs frames into DMA buffers.  The GPU and
  ISP on the Pi 4 natively prefer 4-byte-per-pixel packed formats because
  they are aligned to 32-bit memory bus boundaries.  XRGB8888 is exactly
  this: each pixel is stored as [B, G, R, X] in memory (little-endian
  ARM).  The X byte is a padding byte — its value is undefined (often
  0xFF or 0x00).

  When you request "RGB888" or "BGR888", Picamera2 may silently remap to
  "XRGB8888" on some camera modules (IMX219, OV5647, IMX477) depending on
  the libcamera version shipped with Bookworm.  The resulting array has
  shape (H, W, 4) instead of (H, W, 3).  Applying a 3-channel cvtColor
  to a 4-channel array causes undefined behaviour — black output or
  garbled colours — even though np.mean() appears healthy (the X=0xFF
  byte inflates the mean).

  Fix: request "XRGB8888" explicitly.  capture_array() always returns
  (H, W, 4).  Strip the last channel:  bgr = frame[:, :, :3]
  The first three channels are already in B, G, R order on ARM
  little-endian.  No cvtColor call is needed for OpenCV display.

  Why create_preview_configuration, NOT create_video_configuration
  ─────────────────────────────────────────────────────────────────
  Both configurations work for capture_array() in modern Picamera2
  (>=0.3.12).  However, on Bookworm, "XRGB8888" is only formally
  advertised in the preview formats list of most sensors.  Requesting
  it via create_video_configuration sometimes triggers a libcamera
  assertion or silently renegotiates to a YUV format.  Using
  create_preview_configuration with XRGB8888 always succeeds.

  Why system OpenCV (python3-opencv from apt), NOT pip opencv-python
  ───────────────────────────────────────────────────────────────────
  pip install opencv-python bundles its own GTK3 and libGL shared
  libraries inside the wheel.  On Bookworm, these conflict with the
  system's libGL provided by mesa (required for Wayland / KMS display).
  The symptom: cv2.imshow() opens a window successfully but renders a
  black rectangle — the framebuffer handoff fails silently.
  python3-opencv (apt) is compiled against the exact system GTK/GL
  versions; there is no conflict.

  libcamera STRIDE PADDING
  ─────────────────────────
  libcamera pads each row of a DMA buffer to a multiple of 32 or 64
  bytes for hardware alignment.  For a 640-pixel wide XRGB8888 image,
  each row is 640 * 4 = 2560 bytes — already 64-byte aligned, so no
  padding is added here.  But at other resolutions the stride may be
  larger than width * bytes_per_pixel.  We read the actual stride via
  picam2.stream_configuration("main")["stride"] and verify shape.

  MediaPipe Pose — Classic API
  ─────────────────────────────
  mediapipe.solutions.pose is used (not the Tasks PoseLandmarker) for
  the same reasons as documented in the original design: zero model
  download, simpler setup, battle-tested on aarch64, and equivalent FPS.

  Raised-Arm Detection Algorithm
  ────────────────────────────────
  MediaPipe normalised coordinates: y=0.0 = top, y=1.0 = bottom.
  Therefore wrist.y < shoulder.y means the wrist is physically HIGHER.
  A 0.5-second continuous hold timer eliminates false positives.
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import time
import sys

# ── Third-party: NumPy ────────────────────────────────────────────────────────
import numpy as np

# ── Third-party: OpenCV ───────────────────────────────────────────────────────
# IMPORTANT: cv2 must come from python3-opencv (apt), NOT from pip.
# If you see import errors, run:
#   sudo apt install -y python3-opencv
# and ensure opencv-python is NOT installed via pip in your venv.
try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not available.")
    print("  Install via apt: sudo apt install -y python3-opencv")
    print("  Do NOT use: pip install opencv-python  (causes display conflicts)")
    sys.exit(1)

# ── Third-party: MediaPipe ────────────────────────────────────────────────────
try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] MediaPipe is not installed.")
    print("  Install with: pip install mediapipe>=0.10.0,<0.11.0")
    sys.exit(1)

# ── Third-party: Picamera2 ────────────────────────────────────────────────────
# Must be installed via apt: sudo apt install -y python3-picamera2
# The venv MUST be created with --system-site-packages to access it.
try:
    from picamera2 import Picamera2
except ImportError:
    print("[ERROR] picamera2 is not installed.")
    print("  Install with: sudo apt install -y python3-picamera2")
    print("  Create venv with: python3 -m venv venv --system-site-packages")
    sys.exit(1)

# ── RPi.GPIO — used for hardware PWM control of the MG995 servo ──────────────
# Installed by default on Raspberry Pi OS.  If missing:
#   sudo apt install -y python3-rpi.gpio
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("[ERROR] RPi.GPIO is not available.")
    print("  Install with: sudo apt install -y python3-rpi.gpio")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

# Target capture resolution.  640×480 balances pose accuracy and CPU load.
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# How long (seconds) a wrist must stay above its shoulder before
# "ARM RAISED" fires.  0.5 s eliminates accidental short lifts.
ARM_RAISE_HOLD_SECONDS = 0.5

# MediaPipe Pose model complexity:
#   0 = Lite  (fastest, ~8 FPS on Pi 4, lower accuracy)
#   1 = Full  (balanced, ~6 FPS on Pi 4, good accuracy)  ← chosen
#   2 = Heavy (slowest, ~3 FPS on Pi 4, best accuracy)
POSE_MODEL_COMPLEXITY = 1

# Confidence thresholds for MediaPipe Pose detection and tracking.
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5

# Minimum landmark visibility score to trust a landmark.
# Landmarks below this are treated as occluded / unreliable.
MIN_PROMINENCE_VISIBILITY = 0.6

# Quit key.
EXIT_KEY = ord('q')

# BGR colours for on-screen overlays.
COLOR_GREEN  = (0,   220,  80)
COLOR_RED    = (0,    50, 220)
COLOR_YELLOW = (0,   200, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (230, 200,   0)
COLOR_ORANGE = (0,   165, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  Servo configuration constants
#  All servo tuning values are centralised here for easy adjustment.
# ─────────────────────────────────────────────────────────────────────────────

# GPIO pin (BCM numbering) connected to the MG995 signal wire.
# GPIO18 supports hardware PWM on the Pi 4 — use this pin.
SERVO_GPIO_PIN = 18

# PWM frequency in Hz.  All hobby servos, including MG995, require 50 Hz
# (20 ms period).  Using a different frequency will damage the servo.
SERVO_PWM_FREQ = 50

# Duty-cycle values that correspond to physical servo angles.
# MG995 standard range: 1 ms pulse (0°) to 2 ms pulse (180°).
#   duty = pulse_width_ms / period_ms * 100
#   period_ms = 1000 / 50 Hz = 20 ms
#   0°  → 1.0 ms → 1.0/20.0*100 = 5.0 %
#   90° → 1.5 ms → 1.5/20.0*100 = 7.5 %
# Tune SERVO_DUTY_OPEN / SERVO_DUTY_CLOSED if your servo has a different range.
SERVO_DUTY_CLOSED = 5.0    # duty cycle (%) for lid-closed position (0°)
SERVO_DUTY_OPEN   = 7.5    # duty cycle (%) for lid-open position  (90°)

# Smooth-sweep parameters.
# Number of intermediate steps between CLOSED and OPEN (and back).
# More steps = smoother motion but slightly slower travel.
SERVO_SWEEP_STEPS = 20
# Delay in seconds between each step.  20 steps × 0.015 s ≈ 0.3 s total travel.
SERVO_STEP_DELAY  = 0.015


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Landmark enum shortcuts used by the arm-raise detector.
LEFT_SHOULDER  = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_WRIST     = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST    = mp_pose.PoseLandmark.RIGHT_WRIST


# ─────────────────────────────────────────────────────────────────────────────
#  Camera initialisation — REWRITTEN FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def open_camera():
    """
    Initialise the Raspberry Pi CSI Camera Module using Picamera2.

    Design decisions (see module docstring for full rationale):
      • create_preview_configuration  — advertises XRGB8888 on all sensors
      • format="XRGB8888"            — GPU-native; never silently remapped
      • capture_array("main")        — explicit stream name avoids ambiguity
      • strip [:, :, :3]             — converts (H,W,4) BGRX → (H,W,3) BGR
      • warm-up loop with live check — waits for non-black frames, not just time

    Returns
    -------
    tuple[Picamera2, dict]
        (cam, stream_info) where stream_info contains the actual configured
        format, size, and stride as reported by libcamera after configuration.
    """
    try:
        cam = Picamera2()

        # ── Print sensor info so we know exactly what hardware is attached ────
        sensor_modes = cam.sensor_modes
        model        = cam.camera_properties.get("Model", "Unknown")
        print(f"[INFO] Camera model    : {model}")
        print(f"[INFO] Sensor modes    : {len(sensor_modes)} available")

        # ── Build configuration ───────────────────────────────────────────────
        #
        # XRGB8888 (= BGRX on ARM little-endian) is requested because it is
        # the ISP's native packed output format.  It is listed in the preview
        # formats of every Pi camera module (IMX219, IMX477, OV5647, IMX708).
        # No other format is as universally guaranteed to be accepted.
        #
        # create_preview_configuration is used (not create_video_configuration)
        # because some sensor drivers on Bookworm only advertise XRGB8888 in
        # their preview format list.  Requesting it via the video configuration
        # API can trigger a silent format renegotiation to YUV420.
        #
        # display=None  — we do NOT want a GPU compositor preview window;
        #                 we will display via OpenCV instead.
        # buffer_count=4 — 4 DMA buffers give the ISP headroom to pipeline
        #                  frames without stalling.
        # queue=True    — capture_array() always returns the latest buffered
        #                 frame immediately (non-blocking in steady state).
        #
        config = cam.create_preview_configuration(
            main    = {"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XRGB8888"},
            display = None,
            buffer_count = 4,
            queue   = True,
        )
        cam.configure(config)

        # ── Read back the ACTUAL configured parameters ────────────────────────
        # libcamera may adjust the requested resolution/format for alignment.
        # We must use the ACTUAL values, not the requested ones.
        stream_info = cam.stream_configuration("main")
        actual_fmt    = stream_info["format"]
        actual_size   = stream_info["size"]
        actual_stride = stream_info["stride"]

        print(f"[INFO] Requested format: XRGB8888 {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"[INFO] Actual format   : {actual_fmt} {actual_size[0]}x{actual_size[1]}")
        print(f"[INFO] Actual stride   : {actual_stride} bytes/row")

        # Stride sanity check: for XRGB8888 each pixel is 4 bytes.
        expected_stride = actual_size[0] * 4
        if actual_stride != expected_stride:
            print(
                f"[WARN] Stride padding detected: {actual_stride} vs "
                f"expected {expected_stride}.  Frames will be sliced correctly."
            )

        # ── Start streaming ───────────────────────────────────────────────────
        cam.start()

        # ── Wait for AGC / AWB to converge ───────────────────────────────────
        # Auto-Gain-Control and Auto-White-Balance need several frames to
        # converge from their initial state.  A fixed sleep is NOT sufficient
        # on its own — if the room is dim the first frames may still be black.
        # We therefore loop until we get a frame with measurable brightness.
        print("[INFO] Waiting for camera to produce live frames ...")
        time.sleep(0.5)   # allow the ISP pipeline to flush its startup state

        MAX_WARMUP_ATTEMPTS = 30   # at ~30 FPS this is a 1-second maximum
        live_frame_found    = False

        for attempt in range(1, MAX_WARMUP_ATTEMPTS + 1):
            # capture_array("main") with the explicit stream name is mandatory;
            # without it, older Picamera2 versions grab from an internal default
            # stream that may differ from "main".
            raw = cam.capture_array("main")

            # Strip the XRGB8888 padding byte to get BGR.
            bgr = raw[:, :, :3]

            mean_val = float(np.mean(bgr))
            print(f"[INFO]   Warm-up frame {attempt:>2}/{MAX_WARMUP_ATTEMPTS}: "
                  f"shape={raw.shape}, dtype={raw.dtype}, mean={mean_val:.1f}")

            if mean_val > 2.0:
                # Frame has real pixel data — camera is live.
                print(f"[INFO] Camera is producing live frames (mean={mean_val:.1f}).")
                live_frame_found = True
                break

            time.sleep(0.1)

        if not live_frame_found:
            print(
                "[ERROR] Camera produced only black frames after warm-up.\n"
                "  Possible causes:\n"
                "    1. Lens cap still on or pointing at a very dark surface.\n"
                "    2. CSI ribbon cable loose or inserted incorrectly.\n"
                "    3. Camera module not enabled — run: sudo raspi-config\n"
                "       Interface Options → Camera → Enable → Reboot.\n"
                "    4. Incorrect camera driver — check: libcamera-hello --list-cameras\n"
                "  Program will continue but the preview may remain black."
            )

        return cam, stream_info

    except Exception as exc:
        print(f"[ERROR] Failed to open camera: {exc}")
        print("  Troubleshooting steps:")
        print("  1. Check CSI ribbon cable is firmly seated in the CAMERA port.")
        print("  2. Enable camera: sudo raspi-config → Interface Options → Camera")
        print("  3. Verify detection: libcamera-hello --list-cameras")
        print("  4. Check picamera2: python3 -c \"from picamera2 import Picamera2\"")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Frame acquisition helper — REWRITTEN FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def acquire_bgr_frame(cam, stream_info: dict) -> np.ndarray:
    """
    Capture one frame from the camera and return it as a valid BGR uint8
    numpy array suitable for direct use with cv2.imshow() and cv2.cvtColor().

    The function is format-adaptive: it reads the actual format that libcamera
    negotiated after configuration and converts accordingly.  This avoids the
    silent-format-remap black-frame bug.

    Parameters
    ----------
    cam         : running Picamera2 instance
    stream_info : dict returned by cam.stream_configuration("main")

    Returns
    -------
    np.ndarray  shape=(H, W, 3), dtype=uint8, channels=BGR
    """
    actual_fmt = stream_info["format"]
    W, H       = stream_info["size"]   # libcamera uses (width, height) order

    # Explicit stream name "main" is required.  Without it, some Picamera2
    # versions fall back to an internal default that may point to a different
    # (possibly uninitialised) buffer.
    raw = cam.capture_array("main")

    # ── Format-adaptive conversion ────────────────────────────────────────────
    # XRGB8888 on ARM little-endian is stored in memory as [B, G, R, X] per
    # pixel.  Slicing [:, :, :3] gives [B, G, R] — exactly BGR for OpenCV.
    # The slice is O(0) — it is a NumPy view with no data copy.
    if actual_fmt in ("XRGB8888", "BGRX8888"):
        bgr = raw[:, :, :3]

    # XBGR8888 on ARM little-endian is stored as [R, G, B, X].
    # Slicing gives [R, G, B] = RGB; need a channel flip to BGR.
    elif actual_fmt in ("XBGR8888", "RGBX8888"):
        bgr = raw[:, :, :3]
        bgr = bgr[:, :, ::-1]   # flip R↔B in-place view

    # RGB888: 3 bytes per pixel in [R, G, B] order.
    # Convert to BGR by flipping channels.
    elif actual_fmt == "RGB888":
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    # BGR888: 3 bytes per pixel already in [B, G, R] order.
    elif actual_fmt == "BGR888":
        bgr = raw

    # YUV420: planar YUV.  Y plane is (H, W), UV planes are (H/2, W).
    # Full array shape is (H*1.5, W).  OpenCV handles this directly.
    elif actual_fmt == "YUV420":
        bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_I420)

    else:
        # Unknown format — display as-is and warn.
        print(f"[WARN] Unhandled pixel format '{actual_fmt}'. Displaying raw data.")
        bgr = raw if raw.ndim == 3 and raw.shape[2] == 3 else raw[:, :, :3]

    # ── Stride crop ───────────────────────────────────────────────────────────
    # libcamera pads rows to alignment boundaries.  The returned array may have
    # more columns than requested if stride > width * bytes_per_pixel.
    # Crop to the actual requested resolution so downstream code sees a
    # consistent (FRAME_HEIGHT × FRAME_WIDTH × 3) shape.
    if bgr.shape[1] != FRAME_WIDTH or bgr.shape[0] != FRAME_HEIGHT:
        bgr = bgr[:FRAME_HEIGHT, :FRAME_WIDTH]

    # Ensure contiguous memory layout and uint8 dtype.
    # Some NumPy views (from slicing) are non-contiguous; OpenCV requires
    # C-contiguous arrays for cvtColor and imshow.
    if not bgr.flags["C_CONTIGUOUS"] or bgr.dtype != np.uint8:
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    return bgr


# ─────────────────────────────────────────────────────────────────────────────
#  Servo control helpers
#  These functions are the ONLY place that touches GPIO / PWM.
#  The rest of the file never calls GPIO directly.
# ─────────────────────────────────────────────────────────────────────────────

def init_servo() -> "GPIO.PWM":
    """
    Configure GPIO and create a PWM object for the MG995 servo.

    Uses BCM pin numbering.  Starts the PWM signal at the CLOSED (0°)
    position so the lid is reliably shut when the program launches.

    Returns
    -------
    GPIO.PWM
        The active PWM object.  Keep a reference to it — letting it be
        garbage-collected stops the PWM signal mid-hold.
    """
    GPIO.setmode(GPIO.BCM)          # use BCM (Broadcom) pin numbering
    GPIO.setwarnings(False)         # suppress "channel already in use" noise
    GPIO.setup(SERVO_GPIO_PIN, GPIO.OUT)

    pwm = GPIO.PWM(SERVO_GPIO_PIN, SERVO_PWM_FREQ)
    pwm.start(SERVO_DUTY_CLOSED)    # start at 0° (lid closed)
    time.sleep(0.3)                 # let the servo reach its home position
    
    print(f"[SERVO] Initialised on GPIO{SERVO_GPIO_PIN} @ {SERVO_PWM_FREQ} Hz.")
    print(f"[SERVO] Home position (closed): duty={SERVO_DUTY_CLOSED}% (0°)")
    return pwm


def servo_open(pwm: "GPIO.PWM") -> None:
    """
    Smoothly sweep the servo from the closed position (0°) to the open
    position (90°) — opens the dustbin lid.

    Uses linear interpolation between SERVO_DUTY_CLOSED and SERVO_DUTY_OPEN
    over SERVO_SWEEP_STEPS steps.  This prevents the sudden jolt that
    damages gears when jumping directly to the target duty cycle.

    Parameters
    ----------
    pwm : active GPIO.PWM object created by init_servo()
    """
    print("[SERVO] Opening lid (0° → 90°) ...")
    step_size = (SERVO_DUTY_OPEN - SERVO_DUTY_CLOSED) / SERVO_SWEEP_STEPS
    duty = SERVO_DUTY_CLOSED
    for _ in range(SERVO_SWEEP_STEPS):
        duty += step_size
        pwm.ChangeDutyCycle(duty)
        time.sleep(SERVO_STEP_DELAY)
    # Snap to exact target to correct any floating-point drift.
    pwm.ChangeDutyCycle(SERVO_DUTY_OPEN)
    print(f"[SERVO] Lid open — holding at duty={SERVO_DUTY_OPEN}% (90°).")


def servo_close(pwm: "GPIO.PWM") -> None:
    """
    Smoothly sweep the servo from the open position (90°) back to the
    closed position (0°) — closes the dustbin lid.

    Parameters
    ----------
    pwm : active GPIO.PWM object created by init_servo()
    """
    print("[SERVO] Closing lid (90° → 0°) ...")
    step_size = (SERVO_DUTY_OPEN - SERVO_DUTY_CLOSED) / SERVO_SWEEP_STEPS
    duty = SERVO_DUTY_OPEN
    for _ in range(SERVO_SWEEP_STEPS):
        duty -= step_size
        pwm.ChangeDutyCycle(duty)
        time.sleep(SERVO_STEP_DELAY)
    # Snap to exact home position.
    pwm.ChangeDutyCycle(SERVO_DUTY_CLOSED)
    print(f"[SERVO] Lid closed — duty={SERVO_DUTY_CLOSED}% (0°).")


# ─────────────────────────────────────────────────────────────────────────────
#  Arm-raise detection logic (UNCHANGED from original design)
# ─────────────────────────────────────────────────────────────────────────────

def get_landmark(landmarks, landmark_enum):
    """Return a single NormalizedLandmark from the result list."""
    return landmarks[landmark_enum.value]


def is_arm_raised(landmarks) -> bool:
    """
    Return True if EITHER wrist is currently above its corresponding shoulder.

    MediaPipe normalised y: 0.0 = top of frame, 1.0 = bottom.
    Therefore wrist.y < shoulder.y means the wrist is physically HIGHER
    in the real world (arm raised).

    Only landmarks with sufficient visibility score are trusted so that
    partial occlusion (e.g. person holding trash) does not cause false
    detections.
    """
    l_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    r_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    l_wrist    = get_landmark(landmarks, LEFT_WRIST)
    r_wrist    = get_landmark(landmarks, RIGHT_WRIST)

    left_visible  = (l_shoulder.visibility >= MIN_PROMINENCE_VISIBILITY and
                     l_wrist.visibility    >= MIN_PROMINENCE_VISIBILITY)
    right_visible = (r_shoulder.visibility >= MIN_PROMINENCE_VISIBILITY and
                     r_wrist.visibility    >= MIN_PROMINENCE_VISIBILITY)

    left_raised  = left_visible  and (l_wrist.y < l_shoulder.y)
    right_raised = right_visible and (r_wrist.y < r_shoulder.y)

    return left_raised or right_raised


def select_prominent_person(results):
    """
    Validate that the detected pose belongs to a clearly visible person.

    MediaPipe Pose tracks one person per frame.  This guard prevents the
    detection logic from acting on low-confidence or edge-of-frame detections.
    Returns the landmark list if valid, otherwise None.
    """
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    l_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    r_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)

    # Require at least one shoulder to be clearly visible.
    if (l_shoulder.visibility < MIN_PROMINENCE_VISIBILITY and
            r_shoulder.visibility < MIN_PROMINENCE_VISIBILITY):
        return None

    return landmarks


# ─────────────────────────────────────────────────────────────────────────────
#  HUD drawing helpers (UNCHANGED from original design)
# ─────────────────────────────────────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render FPS counter in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN, 2, cv2.LINE_AA,
    )


def draw_status(frame: np.ndarray, arm_triggered: bool, person_detected: bool) -> None:
    """
    Render detection status in the bottom-left corner.

    READY      — person visible, arm down, waiting for gesture
    ARM RAISED — arm-raise gesture confirmed and held ≥ threshold
    NO PERSON  — no valid pose detected in frame
    """
    if not person_detected:
        label, colour = "NO PERSON",  COLOR_YELLOW
    elif arm_triggered:
        label, colour = "ARM RAISED", COLOR_GREEN
    else:
        label, colour = "READY",      COLOR_WHITE

    cv2.putText(
        frame, label,
        (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA,
    )


def draw_arm_indicator(frame: np.ndarray, landmarks,
                        image_width: int, image_height: int) -> None:
    """
    Draw coloured circles on shoulders and wrists.

    Green = arm raised (wrist above shoulder, landmark visible).
    Red   = arm down (wrist below shoulder, landmark visible).
    Grey  = landmark not sufficiently visible (occluded / out of frame).
    """
    pairs = [
        (LEFT_SHOULDER,  LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_WRIST),
    ]

    for shoulder_lm, wrist_lm in pairs:
        shoulder = get_landmark(landmarks, shoulder_lm)
        wrist    = get_landmark(landmarks, wrist_lm)

        vis_ok = (shoulder.visibility >= MIN_PROMINENCE_VISIBILITY and
                  wrist.visibility    >= MIN_PROMINENCE_VISIBILITY)

        sx = int(shoulder.x * image_width)
        sy = int(shoulder.y * image_height)
        wx = int(wrist.x    * image_width)
        wy = int(wrist.y    * image_height)

        if vis_ok:
            colour = COLOR_GREEN if (wrist.y < shoulder.y) else COLOR_RED
        else:
            colour = (100, 100, 100)

        cv2.circle(frame, (sx, sy), 10, colour, -1)
        cv2.circle(frame, (wx, wy), 10, colour, -1)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main loop — opens the camera, verifies raw preview, then integrates
    MediaPipe Pose for raised-arm detection.

    Pipeline (per frame):
      1. Acquire BGR frame via acquire_bgr_frame()
      2. Display raw BGR frame in a separate diagnostic window
      3. Convert BGR → RGB for MediaPipe (MediaPipe requires RGB input)
      4. Run MediaPipe Pose inference
      5. Draw skeleton + arm indicators on annotated BGR frame
      6. Apply arm-raise detection logic
      7. Overlay HUD (FPS, status) on annotated frame
      8. Display annotated frame
      9. Poll for 'q' key
    """

    # ── Open camera and verify it is producing live frames ───────────────────
    cam, stream_info = open_camera()

    # ── Initialise servo — lid starts in the closed (0°) position ────────────
    # init_servo() configures GPIO BCM, sets up 50 Hz PWM on GPIO18, and
    # moves the servo to SERVO_DUTY_CLOSED before the main loop starts.
    # We keep the pwm object alive for the lifetime of the program.
    servo_pwm = init_servo()

    # ── Print OpenCV / system info for debugging ──────────────────────────────
    print(f"[INFO] OpenCV version  : {cv2.__version__}")
    print(f"[INFO] Build info GUI  : {cv2.getBuildInformation().find('GTK')}")

    # ── MediaPipe Pose model ──────────────────────────────────────────────────
    #
    # static_image_mode=False: video-stream mode; the tracker reuses the
    # previous frame's pose estimate instead of running full detection every
    # frame.  This is 2-3× faster than static_image_mode=True.
    #
    pose = mp_pose.Pose(
        static_image_mode        = False,
        model_complexity         = POSE_MODEL_COMPLEXITY,
        smooth_landmarks         = True,
        min_detection_confidence = POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence  = POSE_TRACKING_CONFIDENCE,
    )

    # ── Arm-raise detection state ─────────────────────────────────────────────
    arm_raise_start      = None   # monotonic time when current raise started
    arm_raised_triggered = False  # True after "ARM RAISED" has been printed

    # ── FPS measurement ───────────────────────────────────────────────────────
    fps            = 0.0
    frame_count    = 0
    fps_start_time = time.monotonic()

    # ── Ongoing frame diagnostics (first N frames) ────────────────────────────
    DIAG_LIMIT = 5
    diag_count = 0

    print("[INFO] Starting arm-raise detection loop. Press 'q' to quit.")
    print("[INFO] Two windows will open:")
    print("[INFO]   • 'Raw Camera Feed' — raw BGR from camera (no MediaPipe)")
    print("[INFO]   • 'Arm Raise Detection' — annotated feed with pose skeleton")
    print("[INFO] If 'Raw Camera Feed' is black, the issue is in camera capture.")
    print("[INFO] If 'Raw Camera Feed' is live but 'Arm Raise Detection' is black,")
    print("[INFO] the issue is in the MediaPipe or display pipeline.")

    try:
        while True:
            # ── 1. Acquire BGR frame ──────────────────────────────────────────
            # acquire_bgr_frame() handles format detection, stride cropping,
            # and contiguous memory layout.  It always returns (H, W, 3) BGR.
            bgr_frame = acquire_bgr_frame(cam, stream_info)

            # ── 2. Per-frame diagnostics (first DIAG_LIMIT frames) ────────────
            if diag_count < DIAG_LIMIT:
                mean_val = float(np.mean(bgr_frame))
                print(
                    f"[DIAG] Frame {diag_count + 1:>2}: "
                    f"shape={bgr_frame.shape}, dtype={bgr_frame.dtype}, "
                    f"mean={mean_val:.1f}"
                )
                if mean_val < 2.0:
                    print(
                        "  [DIAG WARNING] Frame is nearly black (mean < 2.0).\n"
                        "  Check: lens cap, cable seating, room lighting."
                    )
                diag_count += 1
            else:
                # Ongoing silent black-frame detection.
                if float(np.mean(bgr_frame)) < 2.0:
                    print("[WARNING] Black frame received — possible camera dropout.")

            # ── 3. Display RAW frame BEFORE any MediaPipe processing ──────────
            # This window isolates the camera pipeline from MediaPipe.
            # If this window shows a live image, the camera is working.
            cv2.imshow("Raw Camera Feed", bgr_frame)

            # ── 4. FPS measurement ────────────────────────────────────────────
            frame_count += 1
            elapsed = time.monotonic() - fps_start_time
            if elapsed >= 1.0:
                fps            = frame_count / elapsed
                frame_count    = 0
                fps_start_time = time.monotonic()

            # ── 5. Convert BGR → RGB for MediaPipe ───────────────────────────
            # MediaPipe Pose requires an RGB input array.
            # We convert from bgr_frame AFTER displaying it raw so the raw
            # window always shows true camera output.
            # We then work on a separate annotated copy for the main display.
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # Mark non-writeable to skip an internal MediaPipe data copy.
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # ── 6. Build annotated display frame ──────────────────────────────
            # We annotate a COPY of bgr_frame so that the raw window above
            # continues to show unmodified camera output.
            annotated = bgr_frame.copy()

            # Draw the full 33-landmark pose skeleton.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # ── 7. Identify the prominent person ─────────────────────────────
            landmarks       = select_prominent_person(results)
            person_detected = landmarks is not None

            # ── 8. Arm-raise logic ────────────────────────────────────────────
            now = time.monotonic()

            if person_detected:
                currently_raised = is_arm_raised(landmarks)

                if currently_raised:
                    # Start debounce timer on the first frame the arm goes up.
                    if arm_raise_start is None:
                        arm_raise_start = now

                    held = now - arm_raise_start
                    if held >= ARM_RAISE_HOLD_SECONDS and not arm_raised_triggered:
                        # Fire ONCE — the flag prevents repeated servo commands
                        # while the arm remains raised.
                        print("ARM RAISED")
                        servo_open(servo_pwm)   # smoothly rotate 0° → 90°
                        arm_raised_triggered = True

                else:
                    # Arm is down — reset all detection state.
                    if arm_raised_triggered:
                        print("READY")
                        servo_close(servo_pwm)  # smoothly rotate 90° → 0°
                    arm_raise_start      = None
                    arm_raised_triggered = False

                # Draw coloured shoulder/wrist indicators.
                draw_arm_indicator(annotated, landmarks, FRAME_WIDTH, FRAME_HEIGHT)

            else:
                # No valid person detected — close lid if it was open.
                if arm_raised_triggered:
                    print("READY")
                    servo_close(servo_pwm)  # smoothly rotate 90° → 0°
                arm_raise_start      = None
                arm_raised_triggered = False

            # ── 9. Overlay HUD ────────────────────────────────────────────────
            draw_fps(annotated, fps)
            draw_status(annotated, arm_raised_triggered, person_detected)

            # ── 10. Display annotated frame ───────────────────────────────────
            cv2.imshow("Arm Raise Detection", annotated)

            # ── 11. Poll for quit key ─────────────────────────────────────────
            # waitKey(1) keeps latency minimal while pumping the GUI event loop
            # (required for imshow to actually render frames).
            if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                print("[INFO] 'q' pressed — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — exiting.")

    finally:
        # Always release resources, even if an exception occurred.
        print("[INFO] Releasing resources ...")
        pose.close()             # Free MediaPipe model memory
        cam.stop()               # Stop camera streaming
        cam.close()              # Release camera hardware handle
        cv2.destroyAllWindows()  # Close all OpenCV windows
        # ── Servo cleanup ──────────────────────────────────────────────────────
        # Move servo to closed position before stopping PWM so the lid is
        # physically shut when the program exits (not floating mid-travel).
        try:
            servo_pwm.ChangeDutyCycle(SERVO_DUTY_CLOSED)
            time.sleep(0.4)         # let servo reach closed position
            servo_pwm.stop()        # stop PWM signal
        except Exception:
            pass                    # PWM may already be stopped
        GPIO.cleanup()              # release all GPIO pins
        print("[INFO] Done. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry guard
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
