from __future__ import print_function, division
import os
import numpy as np

import config
import models.model as model

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

def get_images(images_dir, transform):
    ''' Import and preprocess image data '''
    from PIL import Image

    X, y = [], []
    v_print('Loading images')
    for ix, fname in enumerate(os.listdir(os.path.abspath(images_dir))):
        # skip hidden files
        if fname.startswith('.'): continue

        # TODO: remove, for testing only
        if ix > 100: break

        # get training correlation (target variable) from filename
        corr = float('.'.join(fname.split('_')[-1].split('.')[0:2]))
        y.append(corr)

        # open and preprocess image, add to dataset
        full_fname = os.path.join(os.path.abspath(images_dir), fname)
        v_print('- Loading image: {}'.format(fname))

        im = Image.open(full_fname).convert('LA')
        im_array = np.array(im.getdata())[:,0].reshape(im.size[0], im.size[1])
        im_array = transform(im_array)
        X.append(im_array)

    v_print('Loaded {} images'.format(len(X)))
    return np.array(X), np.array(y)

def run_training(image_data, image_labels):
    model.build_model(image_data, image_size, args.hsize)

def main():
    args = process_args()

    # set up verbose printing if the arg is included
    _v_print = print if args.verbose else lambda *a, **k: None
    global v_print
    v_print = _v_print

    # preprocessing for images - intensity is all that matters;
    # 1 = black, 0 = white
    # (this could probably just be binary 0/1 for each pixel, grey doesn't
    #  mean anything)
    transform = np.vectorize(lambda x: (config.max_intensity - x) /
                                        config.max_intensity)
    image_data, image_labels = get_images(args.images_dir, transform)

    # TODO: divide into batches
    run_training(image_data, image_labels)


if __name__ == '__main__':
    main()

