import glob
import os
import sys

import numpy as np
import skimage.io
import skimage.transform as trans
directories = glob.glob('data/train/*')
class_names = [os.path.basename(d) for d in directories]
class_names.sort()
num_classes = len(class_names)

paths_train = glob.glob('data/train/*/*')
paths_train.sort()

paths_test = glob.glob('data/test/*')
paths_test.sort()

paths = {
    'train': paths_train,
    'test': paths_test,
}

labels_train = np.zeros(len(paths['train']), dtype='int32')
for k, path in enumerate(paths['train']):
    class_name = os.path.basename(os.path.dirname(path))
    labels_train[k] = class_names.index(class_name)

def load(subset='train', from_s=0, to_s=1):
    """
    Load all images into memory for faster processing
    """
    images = np.empty(len(paths[subset]), dtype='object')
    for k, path in enumerate(paths[subset][from_s:to_s]):
        img = skimage.io.imread(path, as_grey=True)[:, 100:].astype('float32')
        images[k] = img
    
    return images
for i in range(10):
    img = load(from_s=0, to_s=10)[i]
    img = trans.resize(img, (96, 96))
    print(img.dtype)
    print(img.shape)

    skimage.io.imshow(img/255.)
    skimage.io.show()
