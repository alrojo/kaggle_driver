import loader
import utils
import numpy as np 
import pickle as pickle
import sklearn.cross_validation

TARGET_PATH = "./data/validation_split_v1.pkl"

split = sklearn.cross_validation.StratifiedShuffleSplit(loader.labels_train, n_iter=1, test_size=0.1, random_state=np.random.RandomState(42))

indices_train, indices_valid = next(iter(split))

with open(TARGET_PATH, 'bw') as f:
    pickle.dump({
        'indices_train': indices_train,
        'indices_valid': indices_valid,
    }, f)

print("Split stored in %s" % TARGET_PATH)
