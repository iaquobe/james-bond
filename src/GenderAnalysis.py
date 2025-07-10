from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    def __init__(self, scene_directory, analysis_directory, min_confidence=0.5, save_plots=False, batch_size=100):
        # setup directories 
        self.min_confidence     = min_confidence
        self.save_plots         = save_plots
        self.trait_plot_number  = 0
        self.scene_plot_number  = 0
        self.scene_directory    = scene_directory
        self.analysis_directory = analysis_directory
        if not os.path.exists(analysis_directory): 
            os.mkdir(analysis_directory)


        # batches
        self.batch_size = batch_size
        self.batch      = []
        self.scenes     = []


        # CLIP model
        logger.debug("Instanciate clip model")
        self.device                      = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # person traits
        logger.debug("Create trait embeddings")
        self.trait_dict = {
            'gender': ( 
                "woman vs man",
                ['a photo of a woman'],
                ['a photo of a man'],
            ),

            'agency': (
                "submissive vs dominant",
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
                    'a scantily clad person',
                ],[
                    'a professional person',
                    'a person in formal attire',
                    'an adequatly dressed person',
                ],
            ),
        }

        self.trait_labels = [d for (d,_,_) in self.trait_dict.values()]
        self.trait_ranges = {}
        self.traits = []
        for (trait_name, (description, prompts1, prompts2)) in self.trait_dict.items():
            start = len(self.traits)
            self.traits += prompts1
            mid = len(self.traits)
            self.traits += prompts2
            end = len(self.traits)

            self.trait_ranges[trait_name] = (description, start, mid, end)

        self.trait_embeddings = clip.tokenize(self.traits).to(self.device)



    ############################################################################
    ##### Analyze Movies
    ############################################################################
    def analyze_movies(self):
        for movie in os.listdir(self.scene_directory): 
            self.analyze_movie(movie)



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
        if not os.path.exists(self.analysis_directory): 
            os.mkdir(self.analysis_directory)

        # analyze each scene in movie
        yolo    = YOLO("yolo11n.pt")
        results = yolo.predict(input_dir, classes=[0], conf=self.min_confidence, stream=True)
        traits  = []
        for result in results:
            traits.append(self.analyze_scene(result))
        traits.append(self.analyze_batch())


        traits = torch.concat(traits)
        logger.debug(traits)
        movie = {
            'movie_name': movie_name, 
            'traits': traits,
            'trait_description': self.trait_dict,
        }

        out_path = os.path.join(self.analysis_directory, "{}.pkl".format(movie_name))
        with open(out_path, 'wb') as f: 
            pickle.dump(movie, f)



    ############################################################################
    ##### Analyze Scene
    ############################################################################
    def analyze_scene(self, result, force=False): 
        ''' Analyze found persons in image
        Args: 
            result: YOLO result for 
            output_dir (str): 
        '''
        logger.debug("Analyze scene: {}".format(result.path))
        image   = result.orig_img
        persons = result.boxes

        # add all persons to batch
        beg = len(self.batch)
        for person_box in persons: 
            person_image = crop_person(person_box, image)
            self.batch.append(person_image)


        # add original image
        end = len(self.batch)
        if logger.isEnabledFor(logging.DEBUG) and len(persons) > 0:
            cv2_img     = result.plot()
            scene_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            self.plot_scene(cv2_img)
            self.scenes.append((scene_image, beg, end))


        # run if batch full
        if len(self.batch) > self.batch_size or force: 
            return self.analyze_batch()
        return torch.tensor([])


    ############################################################################
    ##### Analyze Persons in Batch
    ############################################################################
    def analyze_batch(self):
        if len(self.batch) == 0: 
            return torch.tensor([])

        # analyze
        batch = torch.stack([self.preprocess(person) for person in self.batch])
        with torch.no_grad():
            logits, _ = self.clip_model(batch, self.trait_embeddings)
            traits = self.compute_batch_traits(logits)

        # plotting
        if logger.isEnabledFor(logging.DEBUG):
            for (scene,beg,end) in self.scenes: 
                for person in range(beg, end): 
                    self.plot_traits(scene, self.batch[person], traits[person])

        # reset
        self.scenes = []
        self.batch = []
        return traits



    def compute_batch_traits(self, logits): 
        res = []
        for (_, (_,beg, mid, end)) in self.trait_ranges.items():
            softmax = torch.stack( [logits[:, beg:mid].mean(dim=1),
                                    logits[:, mid:end].mean(dim=1)]
                                ).T.softmax(dim=1)
            diff = softmax[:,1] - softmax[:,0]
            res.append(diff)
        return torch.stack(res).T

    




    ############################################################################
    ##### Debug Functions 
    ############################################################################
    def plot_scene(self, scene): 
        if self.save_plots: 
            path = os.path.join(self.analysis_directory, 
                                "scene-{:03d}.jpg".format(self.scene_plot_number))
            logger.debug("saving plot to {}".format(path))
            cv2.imwrite(path, scene)
            self.scene_plot_number += 1

    def plot_traits(self, scene, person_image, traits): 
        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 columns

        # Top full-width image (self.current_scene)
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.imshow(scene)
        ax_top.axis("off")
        ax_top.set_title("Scene")

        # Bottom left: person image
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(person_image)
        ax1.axis("off")
        ax1.set_title("Person")

        # Bottom right: bar plot
        values = traits
        logger.debug(self.trait_labels)
        logger.debug(values)

        ax2 = fig.add_subplot(gs[1, 1])
        y_pos = range(len(self.trait_labels))
        ax2.barh(y_pos, values, color='skyblue', height=0.5)
        ax2.axvline(0, color='gray', linewidth=1)
        ax2.set_yticks(y_pos, self.trait_labels)
        ax2.set_xlim(-1, 1)
        ax2.set_xlabel("Spectrum Position")
        ax2.set_title("Opposite Traits Spectrum")

        plt.tight_layout()


        if self.save_plots: 
            path = os.path.join(self.analysis_directory, "traits-{:03d}".format(self.trait_plot_number))
            logger.debug("saving plot to {}".format(path))
            plt.savefig(path)
            self.trait_plot_number += 1
            plt.close()
        else: 
            plt.show()
