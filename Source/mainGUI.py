import sys
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox, QLineEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Importing custom exercise detection functions
from right_curls import detect_right_curls
from left_curls import detect_left_curls
from squats import detect_squats

class PoseEstimationThread(QThread):
    frame_ready = pyqtSignal(QImage)
    finished = pyqtSignal(bool)

    def __init__(self, choice, reps, parent=None):
        super().__init__(parent)
        self.choice = choice
        self.reps = reps
        self.running = True  # Initialize the running state

    def run(self):
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.finished.emit(False)
            return

        reps_completed_right = 0
        reps_completed_left = 0
        reps_completed = 0
        leg_state = "up"
        arm_state_left = "up"
        arm_state_right = "up"
        prev_angle_left = 0
        prev_angle_right = 0

        while self.running:
            success, img = cap.read()
            if not success:
                self.finished.emit(False)
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            try:
                if self.choice == 1:  # Right arm curls
                    reps_completed_right, arm_state_right, prev_angle_right = detect_right_curls(
                        img, results, mpPose, reps_completed_right, self.reps, arm_state_right, prev_angle_right)
                    text = f"Right Reps: {reps_completed_right}/{self.reps}"

                    if reps_completed_right == self.reps:
                        print("Right arm curls completed!")
                        cv2.putText(img, "Right arm curls completed!", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.running = False
                        self.finished.emit(True)

                elif self.choice == 2:  # Left arm curls
                    reps_completed_left, arm_state_left, prev_angle_left = detect_left_curls(
                        img, results, mpPose, reps_completed_left, self.reps, arm_state_left, prev_angle_left)
                    text = f"Left Reps: {reps_completed_left}/{self.reps}"

                    if reps_completed_left == self.reps:
                        print("Left arm curls completed!")
                        cv2.putText(img, "Left arm curls completed!", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.running = False
                        self.finished.emit(True)

                elif self.choice == 3:  # Both arms curls
                    reps_completed_right, arm_state_right, prev_angle_right = detect_right_curls(
                        img, results, mpPose, reps_completed_right, self.reps, arm_state_right, prev_angle_right)
                    reps_completed_left, arm_state_left, prev_angle_left = detect_left_curls(
                        img, results, mpPose, reps_completed_left, self.reps, arm_state_left, prev_angle_left)
                    text = f"Right Reps: {reps_completed_right}/{self.reps}, Left Reps: {reps_completed_left}/{self.reps}"

                    if reps_completed_right == reps_completed_left == self.reps:
                        print("Both arms curls completed!")
                        cv2.putText(img, "Both arms curls completed!", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.running = False
                        self.finished.emit(True)

                elif self.choice == 4:  # Squats
                    reps_completed, leg_state, prev_angle_left, prev_angle_right = detect_squats(
                        img, results, mpPose, reps_completed, self.reps, leg_state, prev_angle_left, prev_angle_right)
                    text = f"Squats: {reps_completed}/{self.reps}"

                    if reps_completed == self.reps:
                        print("Squats completed!")
                        cv2.putText(img, "Squats completed!", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.running = False
                        self.finished.emit(True)

                else:
                    text = "Unknown exercise"

                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert frame to QImage and emit signal
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
                self.frame_ready.emit(qimg)

                # Check for completion
                if (
                        (self.choice in [1, 2, 4] and reps_completed == self.reps) or
                        (self.choice == 3 and reps_completed_right == reps_completed_left == self.reps)
                ):
                    self.running = False
                    self.finished.emit(True)

            except Exception as e:
                print(f"Error in PoseEstimationThread: {e}")
                self.running = False
                break

        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workout Tracker")
        self.setGeometry(100, 100, 800, 600)

        # Apply the style
        self.setStyleSheet("""
            QMainWindow {
                background-color: indigo;  /* Set window background color */
            }
            QLabel {
                color: white;  /* Set text color to white for all labels */
                font-size: 18px;
            }
            QLineEdit {
                color: white;  /* Set text color to white in input field */
                background-color: #4b0082; /* Lighter indigo background for input */
                border: 1px solid white;
                padding: 5px;
            }
            QPushButton {
                color: white;
                background-color: #6a0dad;  /* Lighter indigo background for buttons */
                border-radius: 10px;
                font-size: 16px;
                padding: 10px;
                margin: 5px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Shadow effect */
            }
            QPushButton:hover {
                background-color: #7a1fad;  /* Lighter shade of indigo on hover */
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.4); /* Darker shadow effect on hover */
            }
            QPushButton:pressed {
                background-color: #551a8b;  /* Darker indigo when pressed */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Subtle shadow on button press */
            }
        """)

        # Initialize the window and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Display label for video or messages
        self.video_label = QLabel("Welcome to Workout Tracker", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Input field for repetitions
        self.reps_input_label = QLabel("Enter the number of repetitions:", self)
        self.reps_input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reps_input_label)

        self.reps_input = QLineEdit(self)
        self.reps_input.setPlaceholderText("E.g., 10")
        self.reps_input.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reps_input)

        # Buttons for exercise selection
        self.right_curls_button = QPushButton("Right Arm Curls", self)
        self.right_curls_button.clicked.connect(lambda: self.start_exercise(1))
        layout.addWidget(self.right_curls_button)

        self.left_curls_button = QPushButton("Left Arm Curls", self)
        self.left_curls_button.clicked.connect(lambda: self.start_exercise(2))
        layout.addWidget(self.left_curls_button)

        self.both_curls_button = QPushButton("Both Arms Curls", self)
        self.both_curls_button.clicked.connect(lambda: self.start_exercise(3))
        layout.addWidget(self.both_curls_button)

        self.squats_button = QPushButton("Squats", self)
        self.squats_button.clicked.connect(lambda: self.start_exercise(4))
        layout.addWidget(self.squats_button)

        self.choose_again_button = QPushButton("Choose Again", self)
        self.choose_again_button.clicked.connect(self.reset_ui)
        self.choose_again_button.setVisible(False)
        layout.addWidget(self.choose_again_button)

        self.central_widget.setLayout(layout)

    def start_exercise(self, choice):
        try:
            reps = int(self.reps_input.text())
            self.pose_thread = PoseEstimationThread(choice, reps)
            self.pose_thread.frame_ready.connect(self.update_video_frame)
            self.pose_thread.finished.connect(self.on_exercise_finished)
            self.pose_thread.start()
        except ValueError:
            self.show_message("Please enter a valid number of repetitions.", "Invalid Input")

    def update_video_frame(self, frame):
        pixmap = QPixmap.fromImage(frame)
        self.video_label.setPixmap(pixmap)

    def on_exercise_finished(self, success):
        if success:
            self.show_message("Exercise complete!", "Success")
        else:
            self.show_message("Exercise failed. Try again.", "Error")

        self.choose_again_button.setVisible(True)

    def show_message(self, text, title):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.setStyleSheet("QLabel {color: black;}")  # Change text color to black
        msg.exec_()  # Block until the user presses OK, and the dialog will close after pressing OK

    def reset_ui(self):
    # Clear the video frame and input field
        self.pose_thread.running = False
        self.pose_thread.wait()
        self.video_label.clear()
        self.reps_input.clear()
        self.choose_again_button.setVisible(False)

        # Re-enable buttons
        self.right_curls_button.setEnabled(True)
        self.left_curls_button.setEnabled(True)
        self.both_curls_button.setEnabled(True)
        self.squats_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
