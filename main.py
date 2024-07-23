import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random

video_path = "vehicle.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
model = YOLO('yolov8n.pt')
tracker = Tracker() 
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
cap_out = cv2.VideoWriter('vehicle_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))
detection_threshold = 0.5
unique_ids = set()
vehicles = [1,2,3,5,7]


while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, x2, y1, y2,score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id in vehicles:
                detections.append([x1,x2,y1,y2,score])
        
        tracker.update(frame=frame,detections=detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            unique_ids.add(track_id)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
        
        cv2.putText(frame, f"Vehicle Count: {len(unique_ids)}", (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),1,cv2.LINE_AA)


    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
