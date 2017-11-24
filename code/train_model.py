from __future__ import print_function, division
import os
import numpy as np

import config
import models.mlp as model
import features.featurization as feat

def process_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',
                        default=config.default_batch_size)
    parser.add_argument('-e', '--num-epochs',
                        default=config.default_epochs)
    parser.add_argument('-hs', '--hidden-size',
                        default=config.default_hidden_size)
    parser.add_argument('-i', '--images-dir',
                        default=config.default_images_dir)
    parser.add_argument('-s', '--seed',
                        default=config.default_seed)
    parser.add_argument('-ts', '--test-size',
                        default=config.default_test_size)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-vs', '--validation-size',
                        default=config.default_test_size)
    return parser.parse_args()

def run_training(args, image_data, image_labels):

    for X, y in feat. get_batches(image_data, image_labels, args.batch_size):

    model.build_model(image_data, image_size, args.hsize)

def main():
    args = process_args()

    # set up verbose printing if the arg is included
    _v_print = print if args.verbose else lambda *a, **k: None
    global v_print
    v_print = _v_print

    transform = np.vectorize(lambda x: (config.max_intensity - x) /
                                        config.max_intensity)
    image_data, image_labels = feat.get_images(args.images_dir, transform)

    # TODO: divide into batches
    run_training(args, image_data, image_labels)


if __name__ == '__main__':
    main()

