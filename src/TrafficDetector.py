from ultralytics import YOLO
import cv2
import numpy as np

class TrafficDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.car_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    def get_color(self, img, box):
        """Визначає середній колір автомобіля в межах box. Цей метод сильно варіювався б залежно від вимог, але звачаючи на те, що це тестове завдання я вибрав один із простіших способів."""
        x1, y1, x2, y2 = map(int, box)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return (0, 255, 0)
        
        avg_color = cv2.mean(roi)[:3]
        return tuple(map(int, avg_color))

    def process_frame(self, frame):
        # persistence=True зберігає ID між кадрами
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=self.car_classes,
            verbose=False
        )
        return results[0]
