import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LaneDetection:
    def __init__(self):
        self.height = 480
        self.width = 480
        
    def region_of_interest(self,frame):
        triangle = np.array([[(0,self.height),(self.width//2,self.height//2),(self.width,self.height)]])
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask,triangle,255)
        masked_image = cv2.bitwise_and(frame,mask)
        return masked_image

    def display_lines(self,image,lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        image = cv2.addWeighted(line_image,0.8,image,1,1)
        return image
    
    def lane_detection(self,frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frame,(5,5),0)
        canny = cv2.Canny(blur,50,150)
        cropped_image = self.region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
        return lines

