import os
pj = os.path.join

repo_root = '..'
data_dir = pj(repo_root, 'data')
images_dir = pj(data_dir, 'images')

default_seed = 2
default_batch_size = 10
default_epochs = 20
default_test_size = 0.2
default_hidden_size = 500

max_intensity = 255
