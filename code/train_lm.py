from __future__ import print_function, division
import os
import numpy as np

import config as cfg
import features.featurization as feat

def process_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir',
                        default=cfg.images_dir)
    parser.add_argument('-p', '--patches', action='store_true',
                        help='Preprocess image into patches (basically,\
                              do max pooling before training)')
    parser.add_argument('-r', '--regularizer', default=None,
                        choices=[None, 'l1', 'l2'])
    parser.add_argument('-s', '--seed', type=int,
                        default=cfg.default_seed)
    parser.add_argument('-sk', '--skip_cache', action='store_true')
    parser.add_argument('-t', '--testing', action='store_true')
    parser.add_argument('-ts', '--test_size',
                        default=cfg.default_test_size)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-vs', '--validation_size', type=int,
                        default=cfg.default_test_size)
    return parser.parse_args()

def run_lm(args, image_data, image_labels):
    import sklearn.linear_model as lm
    from sklearn.metrics import mean_squared_error

    X, y = feat.split_dataset(image_data, image_labels,
                              args.validation_size, args.test_size)
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y

    if args.testing:
        X_train = np.concatenate((X_train, X_valid))
        y_train = np.concatenate((y_train, y_valid))
    else:
        X_test = X_valid
        y_test = y_valid

    if args.regularizer is None:
        reg = lm.LinearRegression()
    elif args.regularizer == 'l1':
        reg = lm.LassoCV()
    elif args.regularizer == 'l2':
        reg = lm.RidgeCV()

    reg.fit(X_train, y_train)

    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    print(y_pred_test[:10])
    print(y_test[:10])

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print('Train MSE: {}, validation MSE: {}'.format(train_mse, test_mse))


def main():
    args = process_args()

    cfg.init_verbose_print(args.verbose)

    np.random.seed(args.seed)

    transform = np.vectorize(lambda x: (cfg.max_intensity - x) /
                                        cfg.max_intensity)

    image_data, image_labels = feat.get_images(args.images_dir, transform,
                                               flatten=(not args.patches),
                                               skip_cache=args.skip_cache)
    if args.patches:
        image_data = feat.get_image_patches(image_data, flatten=True)

    run_lm(args, image_data, image_labels)

if __name__ == '__main__':
    main()

