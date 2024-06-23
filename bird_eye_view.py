import cv2
import numpy as np

class BirdsEyeView:
    def __init__(self, width, height, scale=10):
        self.width = width
        self.height = height
        self.scale = scale
        
        # Create a blank canvas for the bird's eye view
        self.bev_width = width
        self.bev_height = height  # Make it the same height as the original frame
        self.bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # Define the transformation matrix
        src = np.float32([[0, height], [width, height], [0, height//2], [width, height//2]])
        dst = np.float32([[0, self.bev_height], [self.bev_width, self.bev_height], 
                          [0, 0], [self.bev_width, 0]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def transform_point(self, x, y):
        pt = np.array([x, y, 1]).reshape(3, 1)
        transformed_pt = self.matrix @ pt
        transformed_pt = transformed_pt / transformed_pt[2]
        return int(transformed_pt[0]), int(transformed_pt[1])

    def add_object(self, x_min, y_min, x_max, y_max, color):
        # Use the bottom center of the bounding box as the object's position
        x = (x_min + x_max) // 2
        y = y_max
        
        bev_x, bev_y = self.transform_point(x, y)
        
        # Draw the object on the bird's eye view
        cv2.circle(self.bev_image, (bev_x, bev_y), 5, color, -1)

    def get_birds_eye_view(self):
        # Add a rectangle to represent the car
        car_width = 50
        car_height = 100
        car_x = (self.bev_width - car_width) // 2
        car_y = self.bev_height - car_height
        cv2.rectangle(self.bev_image, (car_x, car_y), 
                      (car_x + car_width, car_y + car_height), (0, 255, 0), -1)
        
        return self.bev_image

    def reset(self):
        self.bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)