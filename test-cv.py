from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Загружаем модель
model = YOLO("albion_yolo_model7/weights/best.pt")

# Выполняем предсказание на изображении
results = model("image.png",imgsz = 640,conf = 0.5)

# Получаем изображение с наложенными предсказаниями
annotated_img = results[0].plot()

# Показываем через matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("YOLOv8 Detection Result")
plt.show()