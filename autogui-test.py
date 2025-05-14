import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("albion_yolo_model7/weights/best.pt")

while True:
    screenshot = pyautogui.screenshot(region=(0, 0, 1024, 768))  # или координаты окна игры
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pyautogui = RGB → OpenCV = BGR

    result = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    annotated = result.plot()

    cv2.imshow("YOLO Game Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()