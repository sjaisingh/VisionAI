import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. SETTINGS & THRESHOLDS ---
# Eye Aspect Ratio threshold (values below this indicate closed eyes)
EYE_AR_THRESH = 0.22 
# Consecutive frames the eye must be below the threshold to trigger alert
EYE_AR_CONSEC_FRAMES = 15 
# Posture angle threshold (Angle between Ear-Shoulder-Hip)
POSTURE_THRESH = 160 

# --- 2. INITIALIZE MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Indices for Eye Landmarks (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- 3. UTILITY FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_ear(landmarks, eye_indices):
    """Calculates the Eye Aspect Ratio (EAR)."""
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    # Horizontal distance
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h)

# --- 4. MAIN VIDEO LOOP ---
cap = cv2.VideoCapture(0)
counter = 0 # Drowsiness frame counter

print("ErgoVision-AI: Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process Frame
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # --- POSTURE LOGIC ---
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        # Extract coordinates for Ear, Shoulder, and Hip (Left side for profile view)
        ear = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        angle = calculate_angle(ear, shoulder, hip)
        
        # Determine Posture State
        posture_status = "GOOD" if angle > POSTURE_THRESH else "SLOUCHING"
        color = (0, 255, 0) if posture_status == "GOOD" else (0, 0, 255)
        
        cv2.putText(frame, f"Posture: {posture_status} ({int(angle)}deg)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- DROWSINESS LOGIC ---
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        ear_l = get_ear(landmarks, LEFT_EYE)
        ear_r = get_ear(landmarks, RIGHT_EYE)
        avg_ear = (ear_l + ear_r) / 2.0

        if avg_ear < EYE_AR_THRESH:
            counter += 1
            if counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (150, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        else:
            counter = 0

    # Display Feed
    cv2.imshow('ErgoVision-AI Capstone', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
