from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import torch
import clip

import tensorflow

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
results = model.predict("./scenes/indiana", classes=[0], conf=0.6, stream=True)

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




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(im_pil).unsqueeze(0).to(device)
text = clip.tokenize(["a picture of a woman", "a picture of a man"]).to(device)
text = clip.tokenize(["woman", "man", 'fearless', 'afraid', 'strong', 'weak']).to(device)

with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  
print("Label probs:", logits_per_image)  

print("Label probs:", probs)  
print("Label probs:", logits_per_image)  


print(logits_per_image[0][0:2].softmax(dim=-1))
print(logits_per_image[0][2:4].softmax(dim=-1))
print(logits_per_image[0][4:6].softmax(dim=-1))

t = logits_per_image
windows = t.unfold(dimension=1, size=2, step=2)

windows.softmax(dim=2)


logits_per_image[0]


logits_per_image[0][0:2]
logits_per_image[0][2:4]
logits_per_image[0][4:6]


text_prompts = [
    "a woman", "a man",  # gender
    "a strong person", "a weak person"  # strength
]
text = clip.tokenize(text_prompts).to(device)

# One forward pass
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarities = (100.0 * image_features @ text_features.T)

# Show results
for prompt, score in zip(text_prompts, similarities[0]):
    print(f"{prompt}: {score.item():.2f}")



import matplotlib.pyplot as plt
import torch
softmax_values = torch.tensor([0.7, 0.3])  # strong=0.7, weak=0.3

# Calculate position on spectrum: positive = strong, negative = weak
position = softmax_values[0] - softmax_values[1]

plt.figure(figsize=(6, 1.5))

# Plot a horizontal bar centered at 0
plt.barh(0, position, color='skyblue', height=0.5)

# Draw center line at 0 for reference
plt.axvline(0, color='gray', linewidth=1)

# Limits from -1 to 1 for full spectrum
plt.xlim(-1, 1)

# Label ticks
plt.xticks([-1, 0, 1], ['Weak', 'Neutral', 'Strong'])
plt.yticks([])  # no y axis labels needed

plt.title("Strong vs Weak Spectrum")
plt.show()

traits = ["strong", "weak", "man", "woman", "afraid", "brave"]

# Group into tuples of two

print(trait_pairs)



# Trait labels for each pair
traits = [
    ("Strong", "Weak"),
    ("Man", "Woman"),
    ("Afraid", "Brave")
]

windows = t.unfold(dimension=1, size=2, step=2)
softmax_pairs = windows.softmax(dim=2)[0]
positions = softmax_pairs[:, 0] - softmax_pairs[:, 1]
plt.figure(figsize=(8, 3))

y_pos = range(len(traits))
plt.barh(y_pos, positions, color='skyblue', height=0.5)
plt.axvline(0, color='gray', linewidth=1)
plt.yticks(y_pos, [f"{t[0]} vs {t[1]}" for t in traits])
plt.xlim(-1, 1)
plt.xlabel("Spectrum Position")
plt.title("Opposite Traits Spectrum")
plt.tight_layout()
plt.show()


import pickle



traits = [
    # GENDER
    'a photo of a woman',
    'a photo of a man',

    # DEPICTION 
    # control 
    'a photo of a person overwhelmed with the situation',
    'a photo of a person in control of the situation',

    # sexualization
    'a photo of a sexualized person',
    'a photo of a professional person',

    # active 
    'a photo of a person who gets what they want',
    'a photo of a person who is passive',

]
trait_pairs = [(traits[i], traits[i+1]) for i in range(0, len(traits), 2)]

with open('./analysis/charade.pkl', 'rb') as f: 
    loaded = pickle.load(f)


loaded


new = {
    'movie_name': loaded['name'],
    'trait_values': loaded['traits'],
    'trait_pairs': trait_pairs
}

new['trait_pairs']

with open('./analysis/charade.pkl', 'wb') as f: 
    pickle.dump(new, f)



men = [t for t in loaded['trait_values'] if t[0] < 0]
women = [t for t in loaded['trait_values'] if t[0] > 0]


np.array(men).mean(axis=0)





from ultralytics import YOLO
import numpy as np 
import cv2
from PIL import Image

path = "./scenes/1963-with-love-from-russia/1963-with-love-from-russia-Scene-0126-01.jpg"
model = YOLO("yolo11n.pt")
results = model.predict(path, classes=[0], conf=0.5, stream=True)
result = next(results)


result.show()

img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)


im_pil.show()





    


len(prompts)
trait_ranges

import torch
t = torch.tensor(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12]]))



t[0, 0:2].softmax(dim=0)


torch.tensor([t[0, 0:2].mean(), t[0, 2:4].mean()]).softmax(dim=0)

traits = {}
for (dimension, (description, beg, mid, end)) in trait_ranges.items():
    # traits[dimension] = {}
    softmax = torch.tensor([t[0, beg:mid].mean(), t[0, mid:end].mean()]).softmax(dim=0)
    diff = softmax[1] - softmax[0]
    traits[dimension] = {
        'description': description, 
        'value': diff
    }


traits

labels = []
values = []
for (label, value) in traits.items(): 
    labels.append(value["description"]) 
    values.append(value["value"]) 
values = torch.tensor(values)

labels
values




values





################################################################################
### batch processing
################################################################################
from PIL import Image, ImageDraw
import torch
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


path = './scenes/batch/'

text = clip.tokenize(['elefant', 'bird', 'lizard', 'fish', 'bat', 'horse']).to(device)
images = [Image.open(os.path.join(path, f))
          for f in os.listdir(path)]
images += images
images += images
images += images
images += images
images += images
images += images
len(images)

batch = torch.stack([preprocess(image) for image in images])
with torch.no_grad():
    logits_per_image, logits_per_text = model(batch, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()


################################################################################
### without batch processing
################################################################################
sum = []
for image in images: 
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        sum .append(logits_per_image)



t = logits_per_image

for (dimension, (description, beg, mid, end)) in trait_ranges.items():
    print(beg, mid, end)

t



t[:, 0:1].mean(dim=1) 
t[:, 1:2].mean(dim=1)

i = 1
t1 = torch.tensor([t[i, 2:5].mean() ,t[i, 5:8].mean()]).softmax(dim=0)
t1[1] - t1[0]


res = []
for (_, (_,beg, mid, end)) in trait_ranges.items():
    softmax = torch.stack( [t[:, beg:mid].mean(dim=1),
                            t[:, mid:end].mean(dim=1)]
                        ).T.softmax(dim=1)
    diff = softmax[:,1] - softmax[:,0]
    res.append(diff)
traits = torch.stack(res).T





traits = {}
for (dimension, (description, beg, mid, end)) in trait_ranges.items():
    # traits[dimension] = {}
    softmax = torch.tensor([t[0, beg:mid].mean(), t[0, mid:end].mean()]).softmax(dim=0)
    diff = softmax[1] - softmax[0]
    traits[dimension] = {
        'description': description, 
        'value': diff
    }


################################################################################
### trait ranges
################################################################################
trait_dict = {
    'gender': ( 
        "woman vs man",
        ['a photo of a woman'],
        ['a photo of a man'],
    ),

    'agency': (
        "low vs high agency",
        [
            'a person overwhelmed with the situation',
            'a person who is passive',
            'a submissive person',

        ],[
            'a person in control of the situation',
            'a person who gets what they want',
            'a dominant person',
        ],
    ),

    'sexualization' :(
        "sexualized vs professional",
        [
            'a sexualized person',
            'a person in a swimsuit',
        ],[
            'a professional person',
            'a person in a suit',
        ],
    ),
}


trait_ranges = {}
prompts = []
for (trait_name, (description, prompts1, prompts2)) in trait_dict.items():
    start = len(prompts)
    prompts += prompts1
    mid = len(prompts)
    prompts += prompts2
    end = len(prompts)

    trait_ranges[trait_name] = (description, start, mid, end)



images

lookup = {images[0] : "elefant"}

import torch
a = torch.tensor([[0.3, 0.3], [0.2, 0.2]])
b = torch.tensor([[0.1, 0.1]])
b = torch.tensor([])

torch.concat([a,b])






import pickle
with open('./analysis/test/1963-with-love-from-russia.pkl', 'rb') as f: 
    loaded = pickle.load(f)


with open('./analysis/james-bond/1962-dr-no.pkl', 'rb') as f: 
    loaded = pickle.load(f)


import os
path = './analysis/james-bond/'
for file in os.listdir(path): 
    fpath = os.path.join(path,file)
    with open('./analysis/james-bond/1962-dr-no.pkl', 'rb') as f: 
        loaded = pickle.load(f)







a = torch.tensor([[-0.9776, -0.0085, -0.9975], [ 0.8820, -0.1005, -0.9861]])
b = torch.tensor([[-0.9612,  0.2987,  0.2045], [ 0.2783,  0.5522,  0.4185], [ 0.5136,  0.5603,  0.9113], [ 0.5507,  0.5077,  0.7304], [ 0.0326,  0.4548,  0.6571]])
c = torch.tensor([[ 0.9282,  0.2356, -0.9654], [-0.7771,  0.0076, -0.9716]])
d = torch.tensor([[-0.9945,  0.5074, -0.9731], [ 0.8027,  0.4425, -0.3174]])
e = torch.tensor([])



torch.concat([a,b, c, d, e])
torch.concat([a,b, c, d]).shape
