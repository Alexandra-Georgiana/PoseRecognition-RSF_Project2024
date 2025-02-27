import cv2
import mediapipe as mp
from right_curls import detect_right_curls
from left_curls import detect_left_curls
from squats import detect_squats

def get_user_input():
    try:
        print("Select the exercise:")
        print("1. Right arm curls")
        print("2. Left arm curls")
        print("3. Both arms curls")
        print("4. Squats")
        print("Press 'q' to quit")
        choice = int(input("Enter your choice (1/2/3/4): "))
        reps = int(input("Enter the number of repetitions to complete: "))
        if choice not in [1, 2, 3, 4] or reps <= 0:
            raise ValueError("Invalid input. Please try again.")
        return choice, reps
    except ValueError as e:
        print(f"Error: {e}")
        return get_user_input()


def run_pose_estimation(cap, pose, mpDraw, mpPose, choice, reps):
    reps_completed_right = 0
    reps_completed_left = 0
    reps_completed = 0
    arm_state_right = "up"
    arm_state_left = "up"
    prev_angle_right = 0
    prev_angle_left = 0
    leg_state = "up"

    while True:
        try:
            success, img = cap.read()
            if not success:
                break

            # Convert to RGB
            imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imageRGB)

            # Draw landmarks
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # Handle right arm curls
            if choice == 1:
                reps_completed_right, arm_state_right, prev_angle_right = detect_right_curls(
                    img, results, mpPose, reps_completed_right, reps, arm_state_right, prev_angle_right)

                cv2.putText(img, f"Right Reps: {reps_completed_right}/{reps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if reps_completed_right == reps:
                    print("Right arm curls completed!")
                    cv2.putText(img, "Right arm curls completed!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Pose Estimation", img)
                    cv2.waitKey(2000)

                    return True

            # Handle left arm curls
            elif choice == 2:
                reps_completed_left, arm_state_left, prev_angle_left = detect_left_curls(
                    img, results, mpPose, reps_completed_left, reps, arm_state_left, prev_angle_left)

                cv2.putText(img, f"Left Reps: {reps_completed_left}/{reps}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if reps_completed_left == reps:
                    print("Left arm curls completed!")
                    cv2.putText(img, "Left arm curls completed!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Pose Estimation", img)
                    cv2.waitKey(2000)

                    return True

            # Handle both arms curls
            elif choice == 3:
                reps_completed_right, arm_state_right, prev_angle_right = detect_right_curls(
                    img, results, mpPose, reps_completed_right, reps, arm_state_right, prev_angle_right)
                reps_completed_left, arm_state_left, prev_angle_left = detect_left_curls(
                    img, results, mpPose, reps_completed_left, reps, arm_state_left, prev_angle_left)

                # Display rep counts on the image
                cv2.putText(img, f"Right Reps: {reps_completed_right}/{reps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f"Left Reps: {reps_completed_left}/{reps}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if reps_completed_right == reps and reps_completed_left == reps:
                    cv2.putText(img, "Both arms curls completed!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print("Both arms curls completed!")
                    cv2.imshow("Pose Estimation", img)
                    cv2.waitKey(2000)

                    return True

            # Handle squats
            elif choice == 4:
                reps_completed, leg_state, prev_angle_left, prev_angle_right = detect_squats(
                    img, results, mpPose, reps_completed, reps, leg_state, prev_angle_left, prev_angle_right)

                cv2.putText(img, f"Squats: {reps_completed}/{reps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if reps_completed == reps:
                    print("Squats completed!")
                    cv2.putText(img, "Squats completed!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Pose Estimation", img)
                    cv2.waitKey(2000)

                    return True

            # Show the frame
            cv2.imshow("Pose Estimation", img)

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        except Exception as e:
            print(f"Error: {e}")
            break

    return False


if __name__ == "__main__":
    while True:
        try:
            # Get user input for exercise choice and reps
            choice, reps = get_user_input()

            # Initialize MediaPipe Pose
            mpDraw = mp.solutions.drawing_utils
            mpPose = mp.solutions.pose
            pose = mpPose.Pose()

            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Camera could not be opened.")
                exit()

            # Run pose estimation
            if run_pose_estimation(cap, pose, mpDraw, mpPose, choice, reps):
                print("Would you like to do another set?")
                continue_input = input("Enter 'y' to continue or any other key to quit: ").strip().lower()
                if continue_input != 'y':
                    break  # Exit the loop if user does not want to continue
        except Exception as e:
            print(f"Error: {e}")
            break
        cap.release()
        cv2.destroyAllWindows()
