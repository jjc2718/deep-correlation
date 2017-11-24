"""
Code to process and format input data

"""
import numpy as np

def get_images(images_dir, transform):
    ''' Import and preprocess image data '''
    # intensity is all that matters; 1 = black, 0 = white
    # (this could probably just be binary 0/1 for each pixel, grey doesn't
    #  mean anything)
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

def get_batches(X, y, batch_size):
    num_samples = X.shape[0]
    for i in range(0, num_samples, batch_size):
        yield (X[i:i+batch_size], y[i:i+batch_size])

