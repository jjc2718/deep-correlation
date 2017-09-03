"""
Guess the correlation with deep learning!

TODO: document
"""
from __future__ import print_function, division
import os
import numpy as np

DEFAULT_SEED = 2
DEFAULT_BATCH_SIZE = 10
DEFAULT_EPOCHS = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_IMAGES_DIR = '../data/images'
DEFAULT_HIDDEN_SIZE = 500
MAX_INTENSITY = 255

def process_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-e', '--num-epochs', default=DEFAULT_EPOCHS)
    parser.add_argument('-hs', '--hidden-size', default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('-i', '--images-dir', default=DEFAULT_IMAGES_DIR)
    parser.add_argument('-s', '--seed', default=DEFAULT_SEED)
    parser.add_argument('-ts', '--test-size', default=DEFAULT_TEST_SIZE)
    parser.add_argument('-vs', '--validation-size', default=DEFAULT_TEST_SIZE)
    return parser.parse_args()

def get_images(images_dir, transform):
    ''' Import and preprocess image data '''
    from PIL import Image

    X, y = [], []
    for fname in os.listdir(os.path.abspath(images_dir)):
        # skip hidden files
        if fname.startswith('.'): continue

        # get training correlation (target variable) from filename
        corr = float('.'.join(fname.split('_')[-1].split('.')[0:2]))
        y.append(corr)

        # open and preprocess image, add to dataset
        full_fname = os.path.join(os.path.abspath(images_dir), fname)
        im = Image.open(full_fname).convert('LA')
        im_array = np.array(im.getdata())[:,0].reshape(im.size[0], im.size[1])
        im_array = transform(im_array)
        X.append(im_array)

    return np.array(X), np.array(y)


def main():
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten
    from sklearn.model_selection import train_test_split

    args = process_args()

    # preprocessing for images - intensity is all that matters;
    # 1 = black, 0 = white
    # (this could probably just be binary 0/1 for each pixel, grey doesn't
    #  mean anything)
    transform = np.vectorize(lambda x: (MAX_INTENSITY - x) / MAX_INTENSITY)
    X, y = get_images(args.images_dir, transform)
    print('loaded images')

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed)

    # start with a simple single-layer MLP, will try more complicated
    # (convolutional?) models eventually
    inp = Input(shape=(100, 100))
    flat = Flatten()(inp)
    hidden_1 = Dense(args.hidden_size, activation='relu')(flat)
    out = Dense(1, activation='linear')(hidden_1)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mae'])

    print('fitting model')
    model.fit(X_train, y_train, batch_size=args.batch_size,
              epochs=args.num_epochs, verbose=1,
              validation_split=args.validation_size)

    print('evaluating model')
    metrics = model.evaluate(X_test, y_test, verbose=1)
    print(metrics)

if __name__ == '__main__':
    main()
