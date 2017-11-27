"""
Generate labeled training data

"""
import os
import sys
import argparse
import numpy as np
# import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../')
import config as cfg

def generate_corr_pts(num_points, corr=0.8):
    x, y = np.array([0, 1]), np.array([0, 1])
    means = [x.mean(), y.mean()]
    stds = [x.std() / 3, y.std() / 3]
    covs = [[stds[0]**2, stds[0]*stds[1]*corr],
            [stds[0]*stds[1]*corr, stds[1]**2]]
    m = np.random.multivariate_normal(means, covs, 100).T
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num_images', type=int, default=10000)
    p.add_argument('-p', '--num_points', type=int, default=100)
    p.add_argument('-o', '--output_dir', default=config.default_images_dir)
    p.add_argument('-s', '--seed', type=int, default=1)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    cfg.init_verbose_print(args.verbose)

    np.random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.v_print('Writing images to {}'.format(args.output_dir))
    for i in range(args.num_images):
        corr = np.random.randint(1, 100) * 0.01
        pts = generate_corr_pts(args.num_points, corr=corr)
        plt.figure(figsize=(10, 10), dpi=10)
        plt.scatter(*pts)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        fname = os.path.join(args.output_dir,
                             'corr_img_{}_{:.2f}.png'.format(i, corr))
        cfg.v_print('- Writing image {}'.format(fname))
        plt.savefig(fname)
        plt.close()
        plt.clf()

if __name__ == '__main__':
    main()

