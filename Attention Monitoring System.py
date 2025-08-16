import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ----------------------------
# Config & Thresholds
# ----------------------------
@dataclass
class Config:
    ear_thresh: float = 0.21        # < EAR => eye closed
    ear_consec_frames: int = 3      # frames to consider a blink/closure
    yaw_thresh_deg: float = 20.0    # |yaw| > => looking away
    pitch_thresh_deg: float = 20.0  # |pitch| > => looking up/down
    roll_thresh_deg: float = 25.0   # just for display/diagnostics
    gaze_off_center: float = 0.35   # |gaze_x| or |gaze_y| > => off-center
    min_face_conf: float = 0.5


# ----------------------------
# MediaPipe setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# FaceMesh landmark indices used for EAR (per MediaPipe topology)
# Right eye (subject's right): 33, 160, 158, 133, 153, 144
# Left eye  (subject's left) : 362, 385, 387, 263, 373, 380
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]

# Head pose 2D points (approximate corners)
# Choose stable, widely separated points
POSE_LANDMARKS = {
    'nose_tip': 1,   # sometimes 1 or 4 are used; 1 works well in many setups
    'chin': 152,
    'left_eye_outer': 263,
    'right_eye_outer': 33,
    'left_mouth': 291,
    'right_mouth': 61
}

# Simple 3D model reference (in millimeters, rough human face model)
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye outer corner
    [43.3, 32.7, -26.0],    # right eye outer corner
    [-28.9, -28.9, -24.1],  # left mouth corner
    [28.9, -28.9, -24.1]    # right mouth corner
], dtype=np.float64)

# Iris landmark index ranges (require refine_landmarks=True)
RIGHT_IRIS = list(range(468, 473))
LEFT_IRIS = list(range(473, 478))


# ----------------------------
# Utility functions
# ----------------------------

def landmarks_to_np(landmarks, w, h) -> np.ndarray:
    pts = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    ear = (A + B) / (2.0 * C + 1e-6)
    return float(ear)


def get_eye_landmarks(all_pts: np.ndarray, indices: List[int]) -> np.ndarray:
    return all_pts[indices]


def head_pose(w: int, h: int, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build 2D points from selected landmarks
    idx = POSE_LANDMARKS
    image_points = np.array([
        pts[idx['nose_tip']],
        pts[idx['chin']],
        pts[idx['left_eye_outer']],
        pts[idx['right_eye_outer']],
        pts[idx['left_mouth']],
        pts[idx['right_mouth']]
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return np.array([pitch, yaw, roll]), rotation_vec, translation_vec


def iris_center(all_pts: np.ndarray, iris_idx: List[int]) -> Optional[np.ndarray]:
    if len(iris_idx) == 0:
        return None
    iris_pts = all_pts[iris_idx]
    c = iris_pts.mean(axis=0)
    return c


def gaze_vector(eye_pts: np.ndarray, iris_c: np.ndarray) -> Tuple[float, float]:
    # Normalize iris center to eye bounding box center => [-1, 1]
    x_min, y_min = eye_pts.min(axis=0)
    x_max, y_max = eye_pts.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    nx = 0.0 if x_max == x_min else (iris_c[0] - cx) / ((x_max - x_min) / 2.0)
    ny = 0.0 if y_max == y_min else (iris_c[1] - cy) / ((y_max - y_min) / 2.0)
    nx = float(np.clip(nx, -1.5, 1.5))
    ny = float(np.clip(ny, -1.5, 1.5))
    return nx, ny


# ----------------------------
# Attention logic
# ----------------------------
@dataclass
class State:
    closed_counter: int = 0
    blinks: int = 0


def attention_label(cfg: Config, ear_l: float, ear_r: float, yaw: float, pitch: float,
                    gaze: Tuple[float, float]) -> str:
    eyes_open = (ear_l > cfg.ear_thresh) and (ear_r > cfg.ear_thresh)
    looking_forward = (abs(yaw) < cfg.yaw_thresh_deg) and (abs(pitch) < cfg.pitch_thresh_deg)
    gaze_centered = (abs(gaze[0]) < cfg.gaze_off_center) and (abs(gaze[1]) < cfg.gaze_off_center)

    if eyes_open and looking_forward and gaze_centered:
        return 'Attentive'
    if not eyes_open:
        return 'Eyes-Closed'
    if not looking_forward:
        return 'Looking-Away'
    if not gaze_centered:
        return 'Gaze-Off'
    return 'Neutral'


# ----------------------------
# Drawing helpers
# ----------------------------

def draw_overlay(frame, cfg: Config, pts: np.ndarray, ear_l: float, ear_r: float,
                 euler: Optional[np.ndarray], gaze_xy: Tuple[float, float], label: str):
    h, w = frame.shape[:2]

    # Eyes
    cv2.polylines(frame, [get_eye_landmarks(pts, R_EYE)], True, (255, 255, 255), 1)
    cv2.polylines(frame, [get_eye_landmarks(pts, L_EYE)], True, (255, 255, 255), 1)

    # Head pose text
    if euler is not None:
        pitch, yaw, roll = [float(x) for x in euler]
        cv2.putText(frame, f"Pitch:{pitch:5.1f} Yaw:{yaw:5.1f} Roll:{roll:5.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # EAR and gaze
    cv2.putText(frame, f"EAR_L:{ear_l:.3f} EAR_R:{ear_r:.3f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gaze x:{gaze_xy[0]:.2f} y:{gaze_xy[1]:.2f}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Label banner
    color = (0, 200, 0) if label == 'Attentive' else (0, 180, 255) if label == 'Neutral' else (0, 0, 255)
    cv2.rectangle(frame, (0, h - 40), (w, h), color, -1)
    cv2.putText(frame, f"{label}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


# ----------------------------
# Main loop
# ----------------------------

def run(cfg: Config, source: str = "0", csv_path: Optional[str] = None, display: bool = True):
    # Video source
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # CSV logging
    rows = []

    state = State()

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               refine_landmarks=True,  # iris landmarks
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for webcam UX
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            label = 'No-Face'
            ear_l = ear_r = 0.0
            euler = None
            gaze_xy = (0.0, 0.0)

            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                pts = landmarks_to_np(lms, w, h)

                # EAR
                eye_r = get_eye_landmarks(pts, R_EYE)
                eye_l = get_eye_landmarks(pts, L_EYE)
                ear_r = eye_aspect_ratio(eye_r)
                ear_l = eye_aspect_ratio(eye_l)

                # Blink/closure counter
                if ear_l < cfg.ear_thresh and ear_r < cfg.ear_thresh:
                    state.closed_counter += 1
                else:
                    if state.closed_counter >= cfg.ear_consec_frames:
                        state.blinks += 1
                    state.closed_counter = 0

                # Head pose
                euler, rvec, tvec = head_pose(w, h, pts)
                if euler is None:
                    euler = np.array([0.0, 0.0, 0.0])

                # Gaze (iris-based)
                ir_c_r = iris_center(pts, RIGHT_IRIS)
                ir_c_l = iris_center(pts, LEFT_IRIS)
                if ir_c_r is not None and ir_c_l is not None:
                    gx_r, gy_r = gaze_vector(eye_r, ir_c_r)
                    gx_l, gy_l = gaze_vector(eye_l, ir_c_l)
                    gaze_xy = ((gx_r + gx_l) / 2.0, (gy_r + gy_l) / 2.0)

                # Attention label
                pitch, yaw, roll = [float(x) for x in euler]
                label = attention_label(cfg, ear_l, ear_r, yaw, pitch, gaze_xy)

                # Draw overlay
                draw_overlay(frame, cfg, pts, ear_l, ear_r, euler, gaze_xy, label)

            # Append CSV row
            ts = time.time()
            rows.append({
                'timestamp': ts,
                'label': label,
                'ear_left': ear_l,
                'ear_right': ear_r,
                'yaw': float(euler[1]) if euler is not None else np.nan,
                'pitch': float(euler[0]) if euler is not None else np.nan,
                'roll': float(euler[2]) if euler is not None else np.nan,
                'gaze_x': gaze_xy[0],
                'gaze_y': gaze_xy[1]
            })

            if display:
                cv2.imshow('Attention Monitor', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if csv_path:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Student Attention Monitoring — Full Pipeline')
    p.add_argument('--source', default='0', help='0 for webcam, or path to video file')
    p.add_argument('--csv', default=None, help='Optional path to save per-frame metrics CSV')
    p.add_argument('--ear', type=float, default=Config.ear_thresh, help='EAR threshold (default 0.21)')
    p.add_argument('--yaw', type=float, default=Config.yaw_thresh_deg, help='Yaw threshold deg (default 20)')
    p.add_argument('--pitch', type=float, default=Config.pitch_thresh_deg, help='Pitch threshold deg (default 20)')
    p.add_argument('--nogui', action='store_true', help='Disable display/window for headless runs')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(ear_thresh=args.ear, yaw_thresh_deg=args.yaw, pitch_thresh_deg=args.pitch)
    run(cfg, source=args.source, csv_path=args.csv, display=not args.nogui)


if __name__ == '__main__':
    main()
