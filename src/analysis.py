from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import argparse
import torch
import clip
import matplotlib.pyplot as plt
import numpy as np



def box_to_coordinates(box): 
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    return x1, y1, x2, y2



def crop_person(box, image) -> Image.Image : 
    x1,y1, x2,y2 = box_to_coordinates(box)
    img = image[y1:y2, x1:x2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil



def get_features(image_pil, features):
    image = preprocess(image_pil).unsqueeze(0).to(device)
    result = dict()

    for feature, prompt in features.items():
        with torch.no_grad():
            text                = clip.tokenize(prompt).to(device)
            logits_per_image, _ = clip_model(image, text)
            probs               = logits_per_image.softmax(dim=-1).cpu().numpy()

        result[feature] = probs

    return result



def dominant_features(features, person_features): 
    return {feature:features[feature][person_features[feature].argmax()] 
        for feature in features.keys()}




def detect_people(results): 
    for result in results:
        image   = result.orig_img
        persons = result.boxes
        path    = result.path

        for person_box in persons:
            cropped_person = crop_person(person_box, image)
            person_features = get_features(cropped_person, features)


            x1,y1,x2,y2 = box_to_coordinates(person_box)
            print(x2-x1)
            print(y2-y1)

            # cropped_person.show("cropped person")
            # plt.imshow(np.asarray(cropped_person))
            print(person_features)
            print(dominant_features(features, person_features))
            # plt.show()




######################################################################
## MAIN 
######################################################################
# if __name__ == "__main__": 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("scene")

features = {
"Gender"         :["a woman"                      , "a man"                           ],
# "Power"          :["a person in control"          , "a person being controlled"       ],
# "Social class"   :["a wealthy person"             , "a poor person"                   ],
# "Agency"         :["a person taking action"       , "a person waiting to be rescued"  ],
# "Objectification":["a person posed for the camera", "a person doing a task"           ],
# "Violence"       :["a person holding a weapon"    , "a person looking afraid"         ],
# "Authority"      :["a person giving instructions" , "a person receiving instructions" ],
# "Sexualization"  :["a person in a swimsuit"       , "a person in a business suit"     ]
}

# YOLO
yolo    = YOLO("yolo11n.pt")
results = yolo.predict("./scenes/shameless/", classes=[0], conf=0.6, stream=True)

# CLIP
device           = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


result = next(results)



image   = result.orig_img
persons = result.boxes
path    = result.path





persons


type(persons[0])

type(image)

for person_box in persons:
    cropped_person = crop_person(person_box, image)
    person_features = get_features(cropped_person, features)


    x1,y1,x2,y2 = box_to_coordinates(person_box)
    print(x2-x1)
    print(y2-y1)

    # cropped_person.show("cropped person")
    # plt.imshow(np.asarray(cropped_person))
    print(person_features)
    print(dominant_features(features, person_features))
    # plt.show()




import tensorflow as tf

tenser = tf.constant(np.array([[24.2882, 25.2426, 22.5983, 25.1403, 22.9018]])
