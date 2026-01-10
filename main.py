import os
from src.reader import RTSPStreamReader
from src.detector import detect_cars
import cv2  
import time


time_skip = 0.5


if __name__ == "__main__":
    URL = "rtsp://127.0.0.1:8554/live"
    print(f"Запуск тесту потоку: {URL}")
    reader = RTSPStreamReader(URL)
    
    try:
        while True:
            ret, frame = reader.get_frame()
            if ret and frame is not None:
                results = detect_cars(frame)
                for r in results:
                    boxes = r.boxes
                    
                    for box in boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]
                        
                        if label != 'car':
                            continue
                        

                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        conf = box.conf[0]


                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        text = f'{label} {conf:.2f}'
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 0), 2)
                time.sleep(time_skip)


                cv2.imshow("RTSP Reader Test", frame)
            else:
                #print("Очікування кадру...")
                time.sleep(time_skip)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Зупинка користувачем...")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Потік закрито.")