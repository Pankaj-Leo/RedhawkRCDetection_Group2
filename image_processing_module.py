# image_processing_module.py

import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, template_path):
        self.cap = cv2.VideoCapture(0)
        self.template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.akaze = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kp_template, self.desc_template = self.akaze.detectAndCompute(self.template_image, None)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = self.akaze.detectAndCompute(gray_frame, None)
        matches = self.matcher.match(self.desc_template, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Assuming a method to calculate angle and distance here
        angle, distance = self.calculate_angle_and_distance(matches, kp_frame)
        return True, angle, distance

    def calculate_angle_and_distance(self, matches, kp_frame):
        # Dummy implementation, replace with actual calculations
        angle = -10  # Example static value for demonstration
        distance = 800  # Example static value for demonstration
        return angle, distance

    def release(self):
        self.cap.release()
