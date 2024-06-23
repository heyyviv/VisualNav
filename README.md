Dash Cam Visual Odometry with Lane Detection
This repository contains Python code for estimating visual odometry from a dash cam video, including object detection, object tracking, and lane detection using OpenCV and various computer vision techniques.

Overview
Visual odometry is the process of estimating the motion of a camera relative to its surroundings based on sequential images. In this project, we use techniques such as feature detection, descriptor matching, and camera pose estimation to compute the camera's trajectory from a video captured by a dash cam mounted on a car. Additionally, the project includes lane detection to enhance situational awareness and improve motion estimation accuracy.

Features
VisualOdometry Class: Implements a visual odometry pipeline using ORB (Oriented FAST and Rotated BRIEF) for feature detection, brute-force matching for descriptor matching, and RANSAC for robust pose estimation.
ObjectDetection Class: Integrates YOLOv5 for real-time object detection and SORT (Simple Online and Realtime Tracking) for tracking detected objects across frames.
LaneDetection Class: Detects lane lines on the road using Canny edge detection, Hough transform, and geometric transformations for a bird's eye view perspective.
BirdsEyeView Class: Generates a bird's eye view representation of detected objects and lane lines using perspective transformation.
Requirements
Python 3.x
OpenCV (pip install opencv-python)
NumPy (pip install numpy)
Other dependencies as specified in requirements.txt


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/QVjUtX0UvVM/0.jpg)](https://www.youtube.com/watch?v=QVjUtX0UvVM)