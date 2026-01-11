
import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" - I have some issues with this on my machine, not needed for others
import cv2
import time
import argparse
from src.Reader import RTSPStreamReader
from src.TrafficDetector import TrafficDetector

from ultralytics import settings

settings.update({'runs_dir': '.', 'sync': False})

def run_analytics(rtsp_url, timeout=0.1, time_skip=0.5):
    reader = RTSPStreamReader(rtsp_url, timeout=timeout)
    detector = TrafficDetector(plate_model_path="yolov8n.pt")
    
    try:
        while True:
            ret, frame = reader.get_frame()
            if not ret or frame is None:
                continue

            results = detector.process_frame(frame)
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    vehicle_roi = frame[max(0, y1):y2, max(0, x1):x2]
                    
                    plate_text = detector.process_plate(vehicle_roi, obj_id)
                    
                    color = detector.get_color(frame, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"ID: {obj_id} | Plate: {plate_text}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("License Plate Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(time_skip)
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Traffic Analytics System")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="rtsp://127.0.0.1:8554/live",
        help="Path to the RTSP camera stream"
    )
    
    parser.add_argument(
        "--skip", 
        type=float, 
        default=0.05, 
        help="Delay between frames to optimize CPU/GPU load (seconds)"
    )

    parser.add_argument(
        "--timeout", 
        type=float, 
        default=0.1, 
        help="Frame acquisition timeout"
    )

    args = parser.parse_args()

    print(f"Starting stream analysis: {args.source}")
    print("Press 'q' to exit.")

    run_analytics(
        rtsp_url=args.source, 
        timeout=args.timeout, 
        time_skip=args.skip
    )