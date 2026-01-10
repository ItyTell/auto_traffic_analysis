from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def detect_cars(img):
    return model(img) 
