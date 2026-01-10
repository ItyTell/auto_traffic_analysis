from ultralytics import YOLO
import time

model = YOLO("yolo11n.pt")


for i in range(20):
    model.track("rtsp://127.0.0.1:8554/live", show=True) 
    time.sleep(0.5)