import numpy as np
import cv2

def calculate_angle(a, b, c):
    """Calculate angle between two lines."""
    try:
        a = np.array(a)  # Hip
        b = np.array(b)  # Knee
        c = np.array(c)  # Ankle

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except Exception as e:
        print(f"Error in calculate_angle: {e}")

def detect_squats(img, results, mpPose, reps_completed, reps_target, leg_state, prev_angle_left=0, prev_angle_right=0):
    """Detect squats, count repetitions based on up and down motion in both legs simultaneously."""
    try:
        landmarks = results.pose_landmarks.landmark

        #for the left side
        left_hip = landmarks[mpPose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mpPose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mpPose.PoseLandmark.LEFT_ANKLE]

        left_hip_coords = (left_hip.x, left_hip.y, left_hip.z)
        left_knee_coords = (left_knee.x, left_knee.y, left_knee.z)
        left_ankle_coords = (left_ankle.x, left_ankle.y, left_ankle.z)

        angle_left = calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)

        #for the right side
        right_hip = landmarks[mpPose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[mpPose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mpPose.PoseLandmark.RIGHT_ANKLE]

        right_hip_coords = (right_hip.x, right_hip.y, right_hip.z)
        right_knee_coords = (right_knee.x, right_knee.y, right_knee.z)
        right_ankle_coords = (right_ankle.x, right_ankle.y, right_ankle.z)

        angle_right = calculate_angle(right_hip_coords, right_knee_coords, right_ankle_coords)

        # Thresholds for detecting the "up" and "down" motion simultaneously in both legs

        #if both legs are bent
        if angle_left < 100 and angle_right < 100:
            if leg_state != "down":  # Transition from up to down
                leg_state = "down"
                reps_completed += 1
                print(f"Squats completed: {reps_completed}/{reps_target}")

        #if both legs are straight
        elif angle_left > 160 and angle_right > 160:
            if leg_state != "up":  # Transition from down to up
                leg_state = "up"
                # No repetition is counted here; only transition is logged

        prev_angle_left = angle_left  # Update previous angle for the next iteration
        prev_angle_right = angle_right  # Update previous angle for the next iteration

        return reps_completed, leg_state, prev_angle_left, prev_angle_right

    except Exception as e:
        print(f"Error in detect_squats: {e}")

        return reps_completed, leg_state, prev_angle_left, prev_angle_right





