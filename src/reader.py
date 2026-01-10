import cv2
import threading
import time

class RTSPStreamReader:
    def __init__(self, rtsp_url):
        self.stream = cv2.VideoCapture(rtsp_url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
            else:
                self.ret, self.frame = self.stream.read()
            time.sleep(0.01)

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


if __name__ == "__main__":
    URL = "rtsp://localhost:8554/live"
    print(f"Запуск тесту потоку: {URL}")
    reader = RTSPStreamReader(URL)
    
    try:
        while True:
            ret, frame = reader.get_frame()
            
            if ret and frame is not None:
                cv2.imshow("RTSP Reader Test", frame)
            else:
                print("Очікування кадру...")
                time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Зупинка користувачем...")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Потік закрито.")