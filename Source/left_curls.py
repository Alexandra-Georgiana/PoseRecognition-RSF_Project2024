import numpy as np
import cv2

def calculate_angle(a, b, c):
    """Calculate angle between two lines."""
    try:
        a = np.array(a)  # Shoulder
        b = np.array(b)  # Elbow
        c = np.array(c)  # Wrist

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except Exception as e:
        print(f"Error in calculate_angle: {e}")


def detect_left_curls(img, results, mpPose, reps_completed, reps_target, arm_state, prev_angle=0):
    """Detect left arm curls, count repetitions based on up and down motion."""
    try:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mpPose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mpPose.PoseLandmark.LEFT_WRIST]

        # Extract the coordinates (x, y, z) of the landmarks
        shoulder_coords = (left_shoulder.x, left_shoulder.y, left_shoulder.z)
        elbow_coords = (left_elbow.x, left_elbow.y, left_elbow.z)
        wrist_coords = (left_wrist.x, left_wrist.y, left_wrist.z)

        # Calculate the angle of the elbow using the shoulder, elbow, and wrist coordinates
        angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)

        # Thresholds for detecting the "up" and "down" motion
        if angle < 40:  # Angle threshold for fully bent (down position)
            if arm_state != "down":  # Transition from up to down
                arm_state = "down"
                reps_completed += 1
                print(f"Left Reps completed: {reps_completed}/{reps_target}")

        elif angle > 130:  # Angle threshold for fully extended (up position)
            if arm_state != "up":  # Transition from down to up
                arm_state = "up"
                # No repetition is counted here; only transition is logged

        prev_angle = angle  # Update previous angle for the next iteration

        return reps_completed, arm_state, prev_angle

    except Exception as e:
        print(f"Error in detect_left_curls: {e}")

        return reps_completed, arm_state, prev_angle

