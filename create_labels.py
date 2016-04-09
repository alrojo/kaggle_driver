import glob
import os
import sys

import numpy as np

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

print("Saving train labels")
np.save("data/labels_train.npy", labels_train)
print("Gzipping train labels")
os.system("gzip data/labels_train.npy")
