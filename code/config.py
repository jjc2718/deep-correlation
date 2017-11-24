import os
pj = os.path.join

repo_root = '..'
data_dir = pj(repo_root, 'data')
images_dir = pj(data_dir, 'images')

images_data = pj(data_dir, 'ims{}.pkl')

def get_image_data_filename(flatten=False):
    return images_data.format('_flat' if flatten else '')

default_batch_size = 100
default_epochs = 20
default_hidden_size = 500
default_learning_rate = 0.001
default_seed = 2
default_test_size = 0.2

max_intensity = 255

def init_verbose_print(verbose=False):
    """ Set up verbose printing if needed """
    _v_print = print if verbose else lambda *a, **k: None
    global v_print
    v_print = _v_print

