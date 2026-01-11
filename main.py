import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import argparse
from src.reader import RTSPStreamReader
from src.TrafficDetector import TrafficDetector



def run_analytics(rtsp_url, timeout=0.1, time_skip=0.05):
    reader = RTSPStreamReader(rtsp_url, timeout=timeout)
    detector = TrafficDetector()
    
    try:
        while True:
            ret, frame = reader.get_frame()
            if not ret or frame is None:
                continue

            results = detector.process_frame(frame)
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()

                for box, obj_id, conf in zip(boxes, ids, confidences):
                    color = detector.get_color(frame, box)
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"ID: {obj_id} | {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Traffic Analysis (ByteTrack)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(time_skip)
    finally:
        reader.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rtsp://127.0.0.1:8554/live")
    args = parser.parse_args()
    run_analytics(args.source, timeout=0.1 ,time_skip=0.05)