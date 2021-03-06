"""
Code to process and format input data

"""
import os
import numpy as np

import config as cfg

def get_images(images_dir, transform, flatten=False, skip_cache=False):
    ''' Import and preprocess image data '''
    # intensity is all that matters; 1 = black, 0 = white
    # (this could probably just be binary 0/1 for each pixel, grey doesn't
    #  mean anything)

    images_filename = cfg.get_image_data_filename(flatten=flatten)

    if (not skip_cache) and os.path.isfile(images_filename):
        cfg.v_print('Loading image data from cache file: {}'.format(
                    images_filename))
        X, y = get_image_data_from_cache(images_filename)

    else:
        cfg.v_print('Loading image data from directory: {}'.format(
                    images_dir))
        X, y = get_image_data_from_files(images_dir, transform,
                                         flatten=flatten,
                                         skip_cache=skip_cache)

    cfg.v_print('Loaded {} images'.format(len(X)))

    return X, y

def get_image_data_from_cache(images_filename):
    import pickle
    with open(images_filename, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    return X, y

def get_image_data_from_files(images_dir, transform, flatten=False,
                              skip_cache=False):
    from PIL import Image

    X, y = [], []
    for ix, fname in enumerate(os.listdir(os.path.abspath(images_dir))):
        # skip hidden files
        if fname.startswith('.'): continue

        # get training correlation (target variable) from filename
        corr = float('.'.join(fname.split('_')[-1].split('.')[0:2]))
        y.append(corr)

        # open and preprocess image, add to dataset
        full_fname = os.path.join(os.path.abspath(images_dir), fname)
        cfg.v_print('- Loading image: {}'.format(fname))

        im = Image.open(full_fname).convert('LA')
        im_array = np.array(im.getdata())[:,0].reshape(im.size[0], im.size[1])
        im_array = transform(im_array)
        X.append(im_array)

    X = np.array([np.array(r).flatten() for r in X]) if flatten else np.array(X)
    y = np.array(y)

    if not skip_cache:
        import pickle
        images_filename = cfg.get_image_data_filename(flatten=flatten)
        with open(images_filename, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)

    return X, y

def get_image_patches(X, method='max', flatten=False):
    from skimage.measure import block_reduce
    reduce_func = (np.max if method == 'max' else np.mean)
    patches = block_reduce(X, block_size=(1, 5, 5), func=reduce_func)
    if flatten:
        X_patches = np.reshape(patches, (patches.shape[0],
                                         patches.shape[1] * patches.shape[2]))
    else:
        X_patches = patches
    return X_patches

def get_batches(X, y, batch_size):
    num_samples = X.shape[0]
    for i in range(0, num_samples, batch_size):
        yield (X[i:i+batch_size], y[i:i+batch_size])

def split_dataset(X, y, valid_size, test_size):
    from sklearn.model_selection import train_test_split
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_dev, y_dev,
                                                    test_size=test_size)
    return ((X_train, X_valid, X_test),
            (y_train, y_valid, y_test))


