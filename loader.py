
import skimage.io
import skimage.transform
import glob
import pickle

import numpy as np
import os

import utils

directories = glob.glob("data/train/*")
class_names = [os.path.basename(d) for d in directories]
class_names.sort()
num_classes = len(class_names)

DEFAULT_VALIDATION_SPLIT = './data/validation_split_v1.pkl'

paths_train = glob.glob("data/train/*/*")
paths_train.sort()

paths_test = glob.glob("data/test/*")
paths_test.sort()

paths = {
    'train': paths_train,
    'test': paths_test
}

labels_train_path = 'data/labels_train.npy.gz' 
labels_train = utils.load_gz(labels_train_path)

class LoadMethod(object):
    def __init__(self, paths, labels=None):
        self.paths = paths
        self.labels = labels

    def __call__(self, idx):
        if self.labels is not None:
            return skimage.io.imread(self.paths[idx], as_grey=True).astype('float32'), self.labels[idx].astype('int32')
        else:
            return skimage.io.imread(self.paths[idx], as_grey=True).astype('float32')


class SampleGenerator(object):

    def __init__(self, load_method, permutation=None,
        shuffle=False, repeat=False):

        self.load_method = load_method
        self.shuffle = shuffle
        self.repeat = repeat

        if permutation is None:
            self.num_samples = len(self.load_method.paths)
            self.permutation = range(self.num_samples)
        else:
            self.permutation = permutation
            self.num_samples = len(self.permutation)
   
#        print("SampleGenerator initiated")

    def gen_sample(self):
        while True:
            if self.shuffle:
                num_samples = self.num_samples
                self.permutation = np.random.permutation(num_samples)
            for num in range(self.num_samples):
                yield self.load_method(self.permutation[num])
            if not self.repeat:
                break


def preprocess(img):
    img = img[:, 100:]
    img = skimage.transform.resize(img, (96, 96))
    return img

class ChunkGenerator(object):

    def __init__(self, sample_generator, chunk_size, labels=False, patch_size=(96, 96)):
        self.p_x, self.p_y = patch_size
        self.labels = labels
        self.sample_generator = sample_generator
        self.chunk_size = chunk_size
        self.samples = []

    def _make_chunk(self):
        self.chunk = dict()
        c_size = len(self.samples) # might < chunk_size !!!
        assert c_size <= self.chunk_size # can only be smaller
        self.chunk['X'] = np.zeros((c_size, self.p_x, self.p_y, 1), 'float32')
        if self.labels:
            self.chunk['t'] = np.zeros((c_size,), 'int32')
        for idx, sample in enumerate(self.samples):
            if self.labels:
                X, t = sample
            else:
                X = sample
            X = preprocess(X)
            self.chunk['X'][idx, :, :, 0] = X
            if self.labels:
                self.chunk['t'][idx] = t
        return self.chunk

    def gen_chunk(self):
        self.samples = []
        for sample in self.sample_generator.gen_sample():
            self.samples.append(sample)
            if len(self.samples) == self.chunk_size:
                yield self._make_chunk()
                self.samples = []  # reset chunk

        # make a smaller chunk from any remaining samples
        if len(self.samples) > 0:
            yield self._make_chunk()


if __name__ == '__main__':
    load_method = LoadMethod(paths['train'], labels_train)
    split = pickle.load(open(DEFAULT_VALIDATION_SPLIT, 'br'))
    print(split['indices_train'])
    print(split['indices_valid'])
    sample_gen = SampleGenerator(load_method, permutation = split['indices_train'], shuffle=True, repeat=True)
    chunk_gen = ChunkGenerator(sample_gen, chunk_size=64, labels=True)
    for idx, chunk in enumerate(chunk_gen.gen_chunk()):
        if idx == 5:
            for i in range(64):
                skimage.io.imshow(chunk['X'][i, 0, :, :])
                skimage.io.show()
                print(chunk['t'][i])
            assert False
