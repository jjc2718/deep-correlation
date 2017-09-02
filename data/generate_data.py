"""
Generate labeled training data

"""
import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 2
NUM_IMAGES = 100
NUM_POINTS = 100

def generate_corr_pts(num_points=NUM_POINTS, corr=0.8):
    x, y = np.array([0, 1]), np.array([0, 1])
    means = [x.mean(), y.mean()]
    stds = [x.std() / 3, y.std() / 3]
    covs = [[stds[0]**2, stds[0]*stds[1]*corr],
            [stds[0]*stds[1]*corr, stds[1]**2]]
    m = np.random.multivariate_normal(means, covs, 100).T
    return m

def main():

    np.random.seed(SEED)

    if not os.path.exists('./images'):
        os.makedirs('./images')

    for i in range(NUM_IMAGES):
        corr = np.random.randint(1, 100) * 0.01
        pts = generate_corr_pts(corr=corr)
        '''
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        '''
        plt.figure(figsize=(10, 10), dpi=10)
        plt.scatter(*pts)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig('./images/corr_img_{}_{:.2f}.png'.format(i, corr))
        plt.close()
        plt.clf()

if __name__ == '__main__':
    main()

