import mediapipe as mp
import numpy as np
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        # Counters and states
        self.countr = 0
        self.stager = None
        self.repsr = 0

        self.countl = 0
        self.stagel = None
        self.repsl = 0

    def process_pose(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(image_rgb)

    def draw_landmarks(self, frame, results):
        self.mp_draw.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 0, 255)),
            self.mp_draw.DrawingSpec(color=(0, 255, 0))
        )

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def calculate_angle_right(self, results):
        landmarks = results.pose_landmarks.landmark
        rshoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        relbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rwrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        return self.calculate_angle(rshoulder, relbow, rwrist)

    def calculate_angle_left(self, results):
        landmarks = results.pose_landmarks.landmark
        lshoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        lelbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lwrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        return self.calculate_angle(lshoulder, lelbow, lwrist)

    def update_reps_right(self, angle):
        if angle > 140 and self.repsr < 5:
            self.stager = "down"
        if angle < 40 and self.stager == "down":
            self.stager = "up"
            self.countr += 1
            self.repsr += 1

    def update_reps_left(self, angle):
        if angle > 140 and self.repsl < 5 and self.repsr == 5:
            self.stagel = "down"
        if angle < 40 and self.stagel == "down":
            self.stagel = "up"
            self.countl += 1
            self.repsl += 1
