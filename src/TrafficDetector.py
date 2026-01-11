
from ultralytics import YOLO
import cv2
import easyocr

class TrafficDetector:
    def __init__(self, model_path="yolo11n.pt", plate_model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.plate_model = YOLO(plate_model_path)
        
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        self.car_classes = [2, 3, 5, 7] 
        
        self.plate_cache = {}

    def get_color(self, img, box):
        x1, y1, x2, y2 = map(int, box)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return (0, 255, 0)
        avg_color = cv2.mean(roi)[:3]
        return tuple(map(int, avg_color))

    def process_plate(self, vehicle_roi, track_id):
        """Detects and reads the license plate within a vehicle ROI."""
        plate_results = self.plate_model(vehicle_roi, verbose=False)[0]
        
        best_plate_text = None
        max_conf = 0

        for plate in plate_results.boxes.data.tolist():
            px1, py1, px2, py2, p_conf, _ = plate
            
            plate_crop = vehicle_roi[int(py1):int(py2), int(px1):int(px2)]
            if plate_crop.size == 0: continue
            
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, thresh_plate = cv2.threshold(gray_plate, 64, 255, cv2.THRESH_BINARY_INV)

            detections = self.reader.readtext(thresh_plate)
            for (_, text, score) in detections:
                text = text.upper().replace(' ', '')
                if score > max_conf:
                    best_plate_text = text
                    max_conf = score

        if best_plate_text:
            current_cached = self.plate_cache.get(track_id, {"conf": 0})
            if max_conf > current_cached["conf"]:
                self.plate_cache[track_id] = {"plate": best_plate_text, "conf": max_conf}

        return self.plate_cache.get(track_id, {}).get("plate", "Unknown")

    def process_frame(self, frame):
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=self.car_classes,
            verbose=False
        )
        return results[0]