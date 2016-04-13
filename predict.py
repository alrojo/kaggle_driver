import loader
import time
import os
import tensorflow as tf
import importlib
import numpy as np

### THESE VARS SHOULD BE SET IN ANOTHER PLACE
config_name = 'test'
checkpoint_name = 'checkpoint-201'
### SETUP MODEL
chosen_checkpoint = 'train/' + config_name + '/checkpoints/' + checkpoint_name
config_path = 'configurations.' + config_name
predictions_path = 'predictions/' + config_name + '-' + checkpoint_name + '.npy'
config = importlib.import_module(config_path)

print('config_name: \t\t%s' % config_name)
print('chosen_checkpoint: \t%s' % chosen_checkpoint)
print('config_path: \t\t%s' % config_path)
print('predictions_path: \t%s' % predictions_path)

# copy settings that affect the training script
chunk_size = config.Model.chunk_size
name = config.Model.name
Xs = tf.placeholder(tf.float32,   shape=[None, 96, 96, 1], name='X_input')

model = config.Model(
    Xs=Xs)

load_method = dict()
sample_generator = dict()
chunk_generator = dict()

print('making loader ...')
load_method['test'] = loader.LoadMethod(loader.paths['test'])

print('making sample gen ...')
sample_generator['test'] = loader.SampleGenerator(
    load_method['test'],
    shuffle=False,
    repeat=False)

print('making chunk gen ...')
chunk_generator['test'] = loader.ChunkGenerator(
    sample_generator['test'],
    chunk_size=chunk_size,
    labels=False)

def perform_iteration(sess, fetches, feed_dict=None, chunk=None):
    """ Performs one iteration/run.

        Returns tuple containing result and elapsed iteration time.

        Keyword arguments:
        sess:       Tensorflow Session
        fetches:    A single graph element, or a list of graph
                    elements.
        feed_dict:  A dictionary that maps graph elements to values
                    (default: None)
        chunk:      A chunk with data used to fill feed_dict
                    (default: None)
        """
    if not fetches:
        raise ValueError("fetches argument must be a non-empty list")

    if feed_dict is None and chunk is not None:
        feed_dict = {
            Xs:     chunk['X']}

        start_time = time.time()
        res = sess.run(fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

    return (res, elapsed)
print("## INITIATE PREDICTION")
predictions = []
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    saver = tf.train.Saver()
    print('restoring weights ...')
    # restore only if files exist
    saver.restore(sess, chosen_checkpoint)

    print("## PREDICTING...")
    combined_time = 0.0  # total time for each print
    for i, t_chunk in enumerate(chunk_generator['test'].gen_chunk()):

        if i % 100 == 0:
            print('%d: iteration' % i)        
        ## TRAIN START ##
        fetches = [
            model.out_tensor_softmax]
#           summaries,

        res, elapsed_it = perform_iteration(
            sess,
            fetches,
            chunk=t_chunk)
        ## TRAIN END ##
        prediction = res[0]
        predictions.append(prediction)
        combined_time += elapsed_it
predictions = np.concatenate(predictions, axis=0)
np.save(predictions_path, predictions)
