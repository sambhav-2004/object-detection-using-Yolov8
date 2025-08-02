# object detection using YOLO V8
# YOLOv8 Object Detection with OpenCV

This project demonstrates real-time object detection using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model integrated with OpenCV. It utilizes the lightweight `yolov8n` (nano) model for fast and efficient performance on live video or webcam input.

## 🔍 Features

- Real-time object detection
- Webcam or video file input support
- Live visualization with bounding boxes
- Uses `yolov8n.pt` model from Ultralytics

## 🚀 Getting Started

### 📦 Prerequisites

Make sure you have Python 3.8 or higher installed. Then, install the required packages:

pip install torch


#📄 Code Overview
python
Copy
Edit
from ultralytics import YOLO
import cv2

def main():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

   while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

   cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

---
To stop the program press q or use stop button
Let me know if you'd like this in Notion or want a version that logs detections or saves output video.

 
