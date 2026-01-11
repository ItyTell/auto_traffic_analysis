
from ultralytics import YOLO
import cv2
import easyocr
import string


dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    # If the OCR is returning fragments, we should at least ensure 
    # it's not just a single symbol or empty.
    if text is None or len(text) < 2: 
        return False
    return True

def format_license(text):
    """Corrects characters based on their position in the plate."""
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 
        5: dict_int_to_char, 6: dict_int_to_char,
        2: dict_char_to_int, 3: dict_char_to_int
    }
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

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
        """ Took the algorithm from: https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-09-15--yolo8-tracking-and-ocr/2023-09-15/"""
        plate_results = self.plate_model(vehicle_roi, verbose=False)[0]
        
        best_plate_for_frame = None
        highest_conf_for_frame = 0

        for plate in plate_results.boxes.data.tolist():
            px1, py1, px2, py2, p_conf, _ = plate
            plate_crop = vehicle_roi[int(py1):int(py2), int(px1):int(px2)]
            
            if plate_crop.size == 0: continue
            
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            detections = self.reader.readtext(gray) 
            
            full_text = ""
            combined_conf = []

            for (_, text, score) in detections:
                clean_text = text.upper().replace(' ', '').strip()
                if len(clean_text) > 1:
                    full_text += clean_text
                    combined_conf.append(score)

            if full_text:
                avg_conf = sum(combined_conf) / len(combined_conf)
                
                # Apply formatting ONLY if it looks like the right length
                if len(full_text) == 7:
                    full_text = format_license(full_text)

                if avg_conf > highest_conf_for_frame:
                    best_plate_for_frame = full_text
                    highest_conf_for_frame = avg_conf

        # Update Global Cache
        if best_plate_for_frame:
            current_cached = self.plate_cache.get(track_id, {"conf": 0})
            if highest_conf_for_frame > current_cached["conf"]:
                self.plate_cache[track_id] = {
                    "plate": best_plate_for_frame, 
                    "conf": highest_conf_for_frame
                }

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