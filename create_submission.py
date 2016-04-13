import os
import sys

import numpy as np
import pandas as pd

import loader

config_name = 'test'
checkpoint_name = 'checkpoint-201.npy'
predictions_path = 'predictions/' + config_name + '-' + checkpoint_name 
filename = os.path.splitext(os.path.basename(predictions_path))[0]
target_path = 'submissions/%s.csv' % filename

print('config_name: \t\t%s' % config_name)
print('predictions_path: \t%s' % predictions_path)
print('filename: \t\t%s' % filename)

header = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')

print('load predictions ...')
predictions = np.load(predictions_path)

print('generating csv ...')
image_filenames = [os.path.basename(path) for path in loader.paths['test']]
df = pd.DataFrame(predictions, columns=loader.class_names, index=image_filenames)
df.index.name = 'img'
df = df[header]
df.to_csv(target_path)

print('compress with gzip')
os.system('gzip %s' % target_path)

print('  stored in %s.gz' % target_path)
