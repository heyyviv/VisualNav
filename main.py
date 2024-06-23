import cv2 
import numpy as np
from lane_detection import LaneDetection
from object_detection import ObjectDetection
from visual_odometry import VisualOdometry
from bird_eye_view import BirdsEyeView

def show_and_save_video(frames, bev_frames, output_filename='output.mp4'):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_width = frames[0].shape[1] + bev_frames[0].shape[1]
    combined_height = frames[0].shape[0]
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (combined_width, combined_height))

    for frame, bev_frame in zip(frames, bev_frames):
        combined_frame = np.hstack((frame, bev_frame))
        
        # Write the frame to the output video file
        out.write(combined_frame)
        
        cv2.imshow('frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoWriter
    out.release()
width = 480
height = 480
cap = cv2.VideoCapture("data/solidWhiteRight.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
frames = []
while True:
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(height,width))
        frames.append(frame)
    else:
        break
output_image = frames.copy()
Object_Detector = ObjectDetection("model/best.pt")
Object_Detector.detect_and_track(frames[0])

vo = VisualOdometry()
trajectory = []
bev_frames = []
birds_eye = BirdsEyeView(width, height)

for index,frame in enumerate(frames):
    results = Object_Detector.detect_and_track(frame)
    birds_eye.reset()

    # Process frame for visual odometry
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformation = vo.process_frame(gray_frame)
    
    # Extract x and y translation
    x, y = transformation[0, 3], transformation[2, 3]
    trajectory.append((x, y))
    
    if len(results) > 0:
        for result in results:
            color = (255, 0, 0)
            class_id,label, xmin, ymin, xmax, ymax,confidence,object_id = result
            cv2.rectangle(output_image[index],(xmin,ymin),(xmax,ymax),(255,0,0),1)
            cv2.putText(output_image[index], str(object_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            birds_eye.add_object(xmin, ymin, xmax, ymax, color)
        for t in trajectory:
            output_image[index] = vo.draw_trajectory(output_image[index], t[0], t[1], scale=50)
    bev_frame = birds_eye.get_birds_eye_view()
    bev_frames.append(bev_frame)




Lane_Detection = LaneDetection()
lines_ouput = [] 
for frame in frames:
    line = Lane_Detection.lane_detection(frame)
    lines_ouput.append(line)
for i in range(len(output_image)):
    output_image[i] = Lane_Detection.display_lines(output_image[i],lines_ouput[i])



show_and_save_video(output_image,bev_frames)
