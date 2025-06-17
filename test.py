from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw

genders = {"male": 0, "female": 1, "break" : 2}

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
    results = model.predict("./scenes/", classes=[0], conf=0.6, stream=True)

    for result in results:
        b=False
        image = result.orig_img
        persons = result.boxes
        for person in persons: 
            if prompt_gender(image, person) == genders["break"]: 
                return



