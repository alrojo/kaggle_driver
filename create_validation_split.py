import loader
import utils
import numpy as np 
import pickle as pickle
import sklearn.cross_validation
import glob
import pandas as pd

TARGET_PATH = "./data/validation_split_v1.pkl"

driver_list = pd.read_csv('data/driver_imgs_list.csv')
drivers = list(set(driver_list['subject']))
drivers_valid = ['p002', 'p045']
drivers_valid_p = driver_list['subject'].isin(drivers_valid)
base_valid = [img for p, img in zip(list(drivers_valid_p), list(driver_list['img'])) if p]

paths_train = glob.glob('train/*/*')
paths_train.sort()
base_train = [os.path.basename(p) for p in paths_train]

indices_valid = []
indices_train = []
for idx, t in enumerate(base_train):
    for v in base_valid:
        if t == v:
            indices_valid.append(idx)
    if idx not in indices_valid:
        indices_train.append(idx)

indices_train = np.asarray(indices_train)
indices_valid = np.asarray(indices_valid)

#split = sklearn.cross_validation.StratifiedShuffleSplit(loader.labels_train, n_iter=1, test_size=0.1, random_state=np.random.RandomState(42))

#indices_train, indices_valid = next(iter(split))


with open(TARGET_PATH, 'bw') as f:
    pickle.dump({
        'indices_train': indices_train,
        'indices_valid': indices_valid,
    }, f)

print("Split stored in %s" % TARGET_PATH)
