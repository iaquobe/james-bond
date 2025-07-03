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




np.array(women).mean(axis=0)
