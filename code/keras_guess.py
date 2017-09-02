"""
Guess the correlation with deep learning!

TODO: document
"""
from __future__ import print_function, division
import os
import argparse
import numpy as np
# from keras.models import Model

DEFAULT_IMAGES_DIR = '../data/images'
HIDDEN_SIZE = 500
MAX_INTENSITY = 255

num_train, num_test = 80, 20

def get_images(images_dir, transform):
    ''' Import and preprocess image data '''
    from PIL import Image

    X, y = [], []
    for fname in os.listdir(os.path.abspath(args.images_dir)):
        # skip hidden files
        if fname.startswith('.'): continue

        # get training correlation (target variable) from filename
        corr = float('.'.join(fname.split('_')[-1].split('.')[0:2]))
        y.append(corr)

        # open and preprocess image, add to dataset
        full_fname = os.path.join(os.path.abspath(args.images_dir), fname)
        im = Image.open(full_fname).convert('LA')
        im_array = np.array(im.getdata())[:,0].reshape(im.size[0], im.size[1])
        im_array = transform(im_array)
        X.append(im_array)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', default=DEFAULT_IMAGES_DIR)
    args = parser.parse_args()

    # preprocessing for images - intensity is all that matters;
    # 1 = black, 0 = white
    # (this could probably just be binary 0/1 for each pixel, grey doesn't
    #  mean anything)
    transform = np.vectorize(lambda x: (MAX_INTENSITY - x) / MAX_INTENSITY)

    X, y = get_images(args.images_dir, transform)

    print(y[0:5])
    print(X[0:2])
    exit()

    # np.set_printoptions(threshold=np.inf)
    # print(im.format, im.size, im.mode)

    # start with a simple single-layer MLP, will try more complicated
    # (convolutional) models eventually

if __name__ == '__main__':
    main()
