import cv2
import numpy as np

class VideoHandler:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            exit()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return self.rescale_frame(frame)

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def display_angle(self, frame, results, angle, landmark, label):
        point = results.pose_landmarks.landmark[getattr(cv2, f"PoseLandmark.{landmark}.value")]
        coords = np.multiply([point.x, point.y], [frame.shape[1], frame.shape[0]]).astype(int)
        cv2.putText(frame, f"{label}: {int(angle)}", tuple(coords), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def display_curl_status(self, frame, count, reps, label, position):
        status = f"{label} curls: {count}" if reps < 5 else f"{label} curls: {count} Done"
        cv2.putText(frame, status, position, cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def show_frame(self, frame):
        cv2.imshow("Workout Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
