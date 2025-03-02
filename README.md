# PoseRecognition-RSF_Project2024
Workout Tracker
Overview
Workout Tracker is a Python-based application that uses computer vision and machine learning to track and count exercise repetitions. The application supports various exercises such as right arm curls, left arm curls, both arms curls, and squats. It provides real-time feedback and displays the number of completed repetitions.  
Features
Real-time pose estimation using MediaPipe
Supports multiple exercises: right arm curls, left arm curls, both arms curls, and squats
Displays the number of completed repetitions
User-friendly GUI built with PyQt5
Project Structure
Source/mainGUI.py: Main GUI application using PyQt5.
Source/main.py: Command-line interface for selecting and running exercises.
Source/video_handler.py: Handles video capture and frame processing.
Source/right_curls.py: Detects and counts right arm curls.
Source/left_curls.py: Detects and counts left arm curls.
Source/squats.py: Detects and counts squats.
Dependencies
Python 3.8+
OpenCV
MediaPipe
NumPy
PyQt5
