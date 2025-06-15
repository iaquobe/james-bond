from ultralytics import YOLO
import cv2
from PIL import Image
from transformers import pipeline


model = YOLO("yolo11n.pt")
image = cv2.imread("./scenes/Charade (1963)-Scene-0053-01.jpg")
results = model.predict(image)


# persons = [box for box in results[0].boxes if box.cls == 0]
# for person in persons: 
#     x1, y1, x2, y2 = map(int, person.xyxy[0].cpu().numpy())
#     person_image = image[y1:y2, x1:x2]
#
#     face = DeepFace.analyze(person_image, actions=["gender"])
#     face.


persons = [box for box in results[0].boxes if box.cls == 0]
person = persons[0]
x1, y1, x2, y2 = map(int, person.xyxy[0].cpu().numpy())
person_image = image[y1:y2, x1:x2]


classifier = pipeline("image-classification", 
                      model="rizvandwiki/gender-classification")

result = classifier(Image.fromarray(person_image))
print(result)
