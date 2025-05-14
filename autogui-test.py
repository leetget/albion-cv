import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
from time import time
import pygetwindow as gw
try:
    win = gw.getWindowsWithTitle("Albion Online Client")[0]  # Убедись, что название окна точное
except IndexError:
    print("Окно не найдено")
    exit()
left = win.left
top = win.top
width = win.width
height = win.height


model = YOLO("albion_yolo_model7/weights/best.pt")
loop_time  = time()
while True:
    screenshot = pyautogui.screenshot(region=(left, top, width, height))  # или координаты окна игры
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

    result = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    annotated = result.plot()

    cv2.imshow("YOLO Game Detection", annotated)
    print('FPS {}'.format(round(1/(time()-loop_time))))
    loop_time = time()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()