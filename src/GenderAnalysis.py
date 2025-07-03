from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import clip
import cv2
import logging
import pickle
logger = logging.getLogger('james-bond')

################################################################################
##### Util Functions
################################################################################
def crop_person(box, image) -> Image.Image : 
    ''' Returns cropped image of detected person (box)
    Args: 
        box (ultralytics.engine.results.Boxes): box of detected person
        image (cv2.image): scene image
    Returns: 
        PIL.Image: scene image cropped to detected person
    '''
    def box_to_coordinates(box): 
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        return x1, y1, x2, y2

    x1,y1, x2,y2 = box_to_coordinates(box)
    img = image[y1:y2, x1:x2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


################################################################################
##### Gender Analysis
################################################################################
class GenderAnalysis: 
    '''
        analysis movies found in scene_directory
        and outputs analysis to analysis_directory

    '''
    def __init__(self, scene_directory, analysis_directory):
        # setup directories 
        self.scene_directory    = scene_directory
        self.analysis_directory = analysis_directory
        if not os.path.exists(analysis_directory): 
            os.mkdir(analysis_directory)


        # CLIP model
        logger.debug("Instanciate clip model")
        self.device                      = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # person traits
        logger.debug("Create trait embeddings")
        self.traits = [
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
        self.trait_embeddings = clip.tokenize(self.traits).to(self.device)
        self.trait_pairs = [(self.traits[i], self.traits[i+1]) 
                            for i in range(0, len(self.traits), 2)]
        self.clean_trait_pairs = [
            ("woman","man"),
            ("overwhelmed","in control"),
            ("sexualized","professional"),
            ("active","passive"),
        ]



    ############################################################################
    ##### Analyze Movies
    ############################################################################
    def analyze_movies(self):
        pass



    ############################################################################
    ##### Analyze Movie
    ############################################################################
    def analyze_movie(self, movie_name: str):
        ''' Analyze all scene images of a movie. 
        Args: 
            movie_name (str): name of the movie (as found in scene directory)
        '''
        # input/output dirs
        logger.debug("Analyze movie: {}".format(movie_name))
        input_dir  = os.path.join(self.scene_directory   , movie_name)
        output_dir = os.path.join(self.analysis_directory, movie_name)
        if not os.path.exists(output_dir): 
            os.mkdir(output_dir)

        # analyze each scene in movie
        yolo    = YOLO("yolo11n.pt")
        results = yolo.predict(input_dir, classes=[0], conf=0.6, stream=True)
        traits  = []
        for result in results:
            traits += self.analyze_scene(result)

        movie = {
            'movie_name': movie_name, 
            'trait_values': np.array(traits),
            'trait_pairs' : self.trait_pairs,
        }

        out_path = os.path.join(self.analysis_directory, "{}.pkl".format(movie_name))
        with open(out_path, 'wb') as f: 
            pickle.dump(movie, f)



    ############################################################################
    ##### Analyze Scene
    ############################################################################
    def analyze_scene(self, result): 
        ''' Analyze found persons in image
        Args: 
            result: YOLO result for 
            output_dir (str): 
        '''
        logger.debug("Analyze scene: {}".format(result.path))
        image   = result.orig_img
        persons = result.boxes

        if logger.isEnabledFor(logging.DEBUG) and len(persons) > 0:
            self.plot_scene(result)

        traits = []
        for person_box in persons:
            person_image = crop_person(person_box, image)
            traits.append(self.analyze_person(person_image))

        return traits



    ############################################################################
    ##### Analyze Person
    ############################################################################
    def analyze_person(self, person_image): 
        # tokenize person_image
        image = (self.preprocess(person_image)
                    .unsqueeze(0)
                     .to(self.device))

        with torch.no_grad():
            logits, _ = self.clip_model(image, self.trait_embeddings)
            windows         = logits.unfold(dimension=1, size=2, step=2)
            softmax_pairs   = windows.softmax(dim=2)[0]
            traits          = softmax_pairs[:, 1] - softmax_pairs[:, 0]

        if logger.isEnabledFor(logging.DEBUG):
            self.plot_traits(person_image, traits)

        return traits




    ############################################################################
    ##### Debug Functions 
    ############################################################################
    def plot_scene(self, result): 
        result.show()
        cv2.waitKey(0)      # This will block until you press a key or close the window
        cv2.destroyAllWindows()



    def plot_traits(self, person_image, positions): 
        # Plot image and logits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Show image
        ax1.imshow(person_image)
        ax1.axis("off")
        ax1.set_title("Person")

        y_pos = range(len(self.trait_pairs))
        plt.barh(y_pos, positions, color='skyblue', height=0.5)
        ax2.axvline(0, color='gray', linewidth=1)
        ax2.set_yticks(y_pos, [f"{t[0]} vs {t[1]}" for t in self.clean_trait_pairs])
        ax2.set_xlim(-1, 1)
        ax2.set_xlabel("Spectrum Position")
        ax2.set_title("Opposite Traits Spectrum")

        plt.tight_layout()
        plt.show()
