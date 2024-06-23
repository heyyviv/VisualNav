from ultralytics import YOLO
import cv2
import numpy as np
import torch
from yolov5 import YOLOv5
import os
from sort import Sort  # This will now import from the local sort.py file

class ObjectDetection:
    def __init__(self, path):
        self.model = YOLOv5(os.path.join(path), device="cpu")
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def detect_and_track(self, frame):
        results = self.model.predict(frame)
        detections = []
        
        id_to_class = results.__dict__["names"]
        if results.pred[0] is not None and len(results.pred[0]) > 0:
            for *xyxy, conf, cls in results.pred[0]:
                if conf > 0.4:
                    x1, y1, x2, y2 = xyxy
                    detect = [int(x1), int(y1), int(x2), int(y2), float(conf)]
                    detections.append(detect)

        if len(detections) == 0:
            return []

        detections = np.array(detections)
        tracked_objects = self.tracker.update(detections)
        
        result = []
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track[:5]
            bbox = list(map(int, [x1, y1, x2, y2]))
            
            # Find the corresponding detection for this track
            detection_idx = np.argmin(np.sum(np.abs(detections[:, :4] - bbox), axis=1))
            class_id = int(results.pred[0][detection_idx, 5].item())
            class_name = id_to_class[class_id]
            conf = detections[detection_idx, 4]
            
            result.append([class_id, class_name, *bbox, conf, int(track_id)])

        return result

    def draw_tracks(self, frame, tracked_objects):
        for obj in tracked_objects:
            class_id, class_name, x1, y1, x2, y2, conf, track_id = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f} ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame