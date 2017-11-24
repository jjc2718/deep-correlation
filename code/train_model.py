from __future__ import print_function, division
import os
import time
import numpy as np
import tensorflow as tf

import config as cfg
import models.mlp as model
import features.featurization as feat

def process_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int,
                        default=cfg.default_batch_size)
    parser.add_argument('-e', '--num_epochs', type=int,
                        default=cfg.default_epochs)
    parser.add_argument('-hs', '--hidden_size', type=int,
                        default=cfg.default_hidden_size)
    parser.add_argument('-i', '--images_dir',
                        default=cfg.images_dir)
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=cfg.default_learning_rate)
    parser.add_argument('-s', '--seed', type=int,
                        default=cfg.default_seed)
    parser.add_argument('-sk', '--skip_cache', action='store_true')
    parser.add_argument('-ts', '--test_size',
                        default=cfg.default_test_size)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-vs', '--validation_size', type=int,
                        default=cfg.default_test_size)
    return parser.parse_args()

def run_training(args, image_data, image_labels):

    image_size = len(image_data[0])
    num_images = len(image_data)

    imdata = tf.placeholder(tf.float32, shape=(None, image_size))
    labels = tf.placeholder(tf.float32, shape=(None))

    output = model.build_model(imdata, image_size, args.hidden_size)
    loss = model.loss(output, labels)
    train_op = model.run_training(loss, args.learning_rate)

    X, y = feat.split_dataset(image_data, image_labels,
                              args.validation_size, args.test_size)
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(1, args.num_epochs+1):

        for batch_no, (X_batch, y_batch) in enumerate(feat.get_batches(X_train,
                                            y_train,args.batch_size), 1):

            start_time = time.time()

            feed = {
                imdata: X_batch,
                labels: y_batch
            }

            _, loss_value = sess.run([train_op, loss], feed_dict=feed)
            cfg.v_print('Epoch {} ({}/{}): train loss {}'.format(
                epoch,
                (num_images * (epoch - 1)) + (batch_no * args.batch_size),
                num_images * args.num_epochs,
                loss_value))

        feed = {
            imdata: X_valid,
            labels: y_valid
        }
        loss_value = sess.run([loss], feed_dict=feed)

        cfg.v_print('\nValidation loss: {}\n'.format(loss_value[0]))
        time.sleep(2)

def main():
    args = process_args()

    cfg.init_verbose_print(args.verbose)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    transform = np.vectorize(lambda x: (cfg.max_intensity - x) /
                                        cfg.max_intensity)
    image_data, image_labels = feat.get_images(args.images_dir, transform,
                                               flatten=True,
                                               skip_cache=args.skip_cache)

    run_training(args, image_data, image_labels)


if __name__ == '__main__':
    main()

