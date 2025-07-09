import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import numpy as np
import logging
import torch
import pickle
logger = logging.getLogger('james-bond')


def parse_args(): 
    # create parser
    parser = argparse.ArgumentParser('plot-results')
    parser.add_argument('analysis_dir')
    parser.add_argument('-d', '--debug', action='store_true')

    # parse arguments 
    args = parser.parse_args()
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



with open("analysis/test/1963-with-love-from-russia.pkl", "rb") as f: 
    love_russia = pickle.load(f)


with open("analysis/test/1964-goldfinger.pkl", "rb") as f: 
    goldfinger = pickle.load(f)

