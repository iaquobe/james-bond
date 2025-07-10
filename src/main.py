import argparse
from os.path import exists, join
import sys
import os
import logging


def parse_args(): 
    # create parser
    parser = argparse.ArgumentParser('gender-analysis')
    parser.add_argument('scene_directory')
    parser.add_argument('output_dir')
    parser.add_argument('-m', '--movie-name', type=str, default=argparse.SUPPRESS)
    parser.add_argument('-s', '--scene-name', type=str, default=argparse.SUPPRESS)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--save-plots', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=100)

    # parse arguments 
    args = parser.parse_args()
    if not os.path.exists(args.scene_directory):
        sys.exit("{} does not exist".format(args.scene_directory))

    # if movie set check that it exists 
    if 'movie_name' in args:
        movie_path = os.path.join(args.scene_directory, args.movie_name)
        if not os.path.exists(movie_path): 
            sys.exit("movie {} does not exist".format(movie_path))

    # if scene set check that it exists 
    if 'scene_name' in args:
        if not os.path.exists(args.scene_name): 
            sys.exit("scene {} does not exist".format(args.scene_name))

    return args



if __name__ == '__main__': 
    args = parse_args()

    # set logging if debug set 
    logger = logging.getLogger('james-bond')
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        print('enable debug')
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # instanciate analysis class
    logger.debug("instanciate class")
    from GenderAnalysis import GenderAnalysis
    analysis = GenderAnalysis(args.scene_directory, args.output_dir, save_plots=args.save_plots, batch_size=args.batch_size)

    # movie name set
    if 'movie_name' in args:
        logger.debug("Analyzing movie")
        analysis.analyze_movie(args.movie_name)

    # scene set
    elif 'scene_name' in args: 
        logger.debug("Analyzing scene")
        from ultralytics import YOLO
        yolo    = YOLO("yolo11n.pt")
        results = yolo.predict(args.scene_name, classes=[0])

        result = results[0]
        res = analysis.analyze_scene(result, force=True)
        logger.debug(res)

    # complete analysis
    else: 
        logger.debug("Analyzing movies")
        analysis.analyze_movies()

