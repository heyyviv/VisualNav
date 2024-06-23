import cv2
import numpy as np

class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_frame = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

    def process_frame(self, frame):
        if self.last_frame is None:
            self.last_frame = frame
            return np.eye(4)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(self.last_frame, None)
        kp2, des2 = self.orb.detectAndCompute(frame, None)

        # Match descriptors
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

        # Update current position
        self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)

        # Update last frame
        self.last_frame = frame

        # Return transformation matrix
        return np.vstack((np.hstack((self.cur_R, self.cur_t)), [0, 0, 0, 1]))

    def draw_trajectory(self, frame, x, y, scale=1):
        h, w = frame.shape[:2]
        x = int(x * scale) + w // 2
        y = int(y * scale) + h // 2
        cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)
        return frame