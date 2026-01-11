import cv2
import threading
import time

class RTSPStreamReader:
    def __init__(self, rtsp_url, timeout=0.5):
        self.stream = cv2.VideoCapture(rtsp_url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.timeout = timeout

    def update(self):
        while not self.stopped:
            if not self.stream or not self.stream.isOpened():
                time.sleep(0.5)
                continue
                
            try:
                if self.stream.grab():
                    ret, frame = self.stream.retrieve()
                    if ret:
                        self.frame = frame
                        self.ret = True
                else:
                    time.sleep(0.1)
                    
            except cv2.error as e:
                #print(f"OpenCV Error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Reading error: {e}")
                break

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()


if __name__ == "__main__":
    URL = "rtsp://localhost:8554/live"
    print(f"Testing the stream: {URL}")
    reader = RTSPStreamReader(URL)
    
    try:
        while True:
            ret, frame = reader.get_frame()
            
            if ret and frame is not None:
                cv2.imshow("RTSP Reader Test", frame)
            else:
                print("Wating for the stream...")
                time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopped by the user...")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("The stream has been closed.")