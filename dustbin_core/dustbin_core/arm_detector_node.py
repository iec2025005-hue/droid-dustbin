#!/usr/bin/env python3
import time
import sys
import numpy as np
import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    from picamera2 import Picamera2
except ImportError:
    pass

# --- Configuration Constants ---
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
ARM_RAISE_HOLD_SECONDS = 0.5
POSE_MODEL_COMPLEXITY = 1
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5
MIN_PROMINENCE_VISIBILITY = 0.6

COLOR_GREEN  = (0,   220,  80)
COLOR_RED    = (0,    50, 220)
COLOR_YELLOW = (0,   200, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (230, 200,   0)

try:
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_styles

LEFT_SHOULDER  = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_WRIST     = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST    = mp_pose.PoseLandmark.RIGHT_WRIST

def open_camera():
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main    = {"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XRGB8888"},
        display = None,
        buffer_count = 4,
        queue   = True,
    )
    cam.configure(config)
    stream_info = cam.stream_configuration("main")
    cam.start()
    time.sleep(0.5)
    return cam, stream_info

def acquire_bgr_frame(cam, stream_info):
    raw = cam.capture_array("main")
    actual_fmt = stream_info["format"]
    
    if actual_fmt in ("XRGB8888", "BGRX8888"):
        bgr = raw[:, :, :3]
    elif actual_fmt in ("XBGR8888", "RGBX8888"):
        bgr = raw[:, :, :3][:, :, ::-1]
    elif actual_fmt == "RGB888":
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    elif actual_fmt == "BGR888":
        bgr = raw
    elif actual_fmt == "YUV420":
        bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_I420)
    else:
        bgr = raw[:, :, :3]

    if bgr.shape[1] != FRAME_WIDTH or bgr.shape[0] != FRAME_HEIGHT:
        bgr = bgr[:FRAME_HEIGHT, :FRAME_WIDTH]

    if not bgr.flags["C_CONTIGUOUS"] or bgr.dtype != np.uint8:
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    return bgr

def get_landmark(landmarks, landmark_enum):
    return landmarks[landmark_enum.value]

def is_arm_raised(landmarks):
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
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    l_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    r_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    if (l_shoulder.visibility < MIN_PROMINENCE_VISIBILITY and
            r_shoulder.visibility < MIN_PROMINENCE_VISIBILITY):
        return None
    return landmarks

class ArmDetectorNode(Node):
    def __init__(self):
        super().__init__('arm_detector_node')
        self.publisher_ = self.create_publisher(Bool, '/dustbin/arm_raised', 10)
        
        self.get_logger().info("Initialising Camera...")
        self.cam, self.stream_info = open_camera()
        
        self.pose = mp_pose.Pose(
            static_image_mode        = False,
            model_complexity         = POSE_MODEL_COMPLEXITY,
            smooth_landmarks         = True,
            min_detection_confidence = POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence  = POSE_TRACKING_CONFIDENCE,
        )

        self.arm_raise_start = None
        self.arm_raised_triggered = False

        self.fps_start_time = time.monotonic()
        self.frame_count = 0
        self.fps = 0.0

        # Run capture loop at ~30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("Arm Detector Node Started")

    def timer_callback(self):
        bgr_frame = acquire_bgr_frame(self.cam, self.stream_info)
        
        # FPS
        self.frame_count += 1
        now = time.monotonic()
        if now - self.fps_start_time >= 1.0:
            self.fps = self.frame_count / (now - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = now

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        annotated = bgr_frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        landmarks = select_prominent_person(results)
        person_detected = landmarks is not None

        if person_detected:
            currently_raised = is_arm_raised(landmarks)
            if currently_raised:
                if self.arm_raise_start is None:
                    self.arm_raise_start = now
                held = now - self.arm_raise_start
                if held >= ARM_RAISE_HOLD_SECONDS and not self.arm_raised_triggered:
                    self.arm_raised_triggered = True
                    self.publish_state(True)
            else:
                if self.arm_raised_triggered:
                    self.publish_state(False)
                self.arm_raise_start = None
                self.arm_raised_triggered = False
        else:
            if self.arm_raised_triggered:
                self.publish_state(False)
            self.arm_raise_start = None
            self.arm_raised_triggered = False

        # Status overlay
        label = "READY" if person_detected else "NO PERSON"
        if self.arm_raised_triggered: label = "ARM RAISED"
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN, 2)
        cv2.putText(annotated, label, (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        cv2.imshow("Arm Raise Detection ROS2", annotated)
        cv2.waitKey(1)

    def publish_state(self, state: bool):
        msg = Bool()
        msg.data = state
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published /dustbin/arm_raised: {state}")

    def cleanup(self):
        self.pose.close()
        self.cam.stop()
        self.cam.close()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ArmDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
