from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import torch
import clip

genders = {"male": 0, "female": 1, "break" : 2}
features = {"wealth": ["a rich person", "a poor person"]}

def box_to_coordinates(box): 
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    return (x1, y1), (x2, y2)


def prompt_gender(image, person): 
    # draw rectangle
    p1, p2 = box_to_coordinates(person)
    person_img = image.copy()
    cv2.rectangle(person_img, p1, p2, (0,255,0), 2)
    cv2.imshow("Person", person_img)

    # get user class
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            cv2.destroyAllWindows()
            return genders["male"]
        if key == ord('m'):
            cv2.destroyAllWindows()
            return genders["male"]
        if key == ord('q'):
            cv2.destroyAllWindows()
            return genders["break"]




def annotate(): 
    model = YOLO("yolo11n.pt")
    results = model.predict("./scenes/charade/", classes=[0], conf=0.6, stream=True)

    for result in results:
        image   = result.orig_img
        persons = result.boxes
        path    = result.path

        for person in persons: 
            if prompt_gender(image, person) == genders["break"]: 
                return



annotate()



model = YOLO("yolo11n.pt")
results = model.predict("./scenes/charade/", classes=[0], conf=0.6, stream=True)

result = next(results)
while len(result.boxes) == 0: 
    result = next(results)

(x1,y1), (x2,y2) = box_to_coordinates(result.boxes[0])
img = result.orig_img[y1:y2, x1:x2]

cv2.imshow("cropped", img)
cv2.waitKey()
cv2.destroyAllWindows()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)







result.path.split(".")[0] + ".pred"
result.path.split(".")[0] + ".gender"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(im_pil).unsqueeze(0).to(device)
text = clip.tokenize(["a woman", "a man"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  


text = clip.tokenize(["a rich person", "a poor person"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  

text = clip.tokenize(["a woman", "a man"]).to(device)

with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  

