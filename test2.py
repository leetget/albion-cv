from ultralytics import YOLO

# Загружаем уже обученную модель
model = YOLO("albion_yolo_model7/weights/best.pt")

# Предсказание на изображении
model.predict("image.png", save=True,imgsz = 640,conf = 0.1)
#model.predict(source=0,show = True,imgsz = 640,conf =)