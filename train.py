
import loader
import time
import click
import os
import numpy as np
import tensorflow as tf
from warnings import warn
import importlib
import pickle


from model import Model
import utils #from utils import basic as utils

SAVER_FOLDER_PATH = {'base': 'train/',
                     'checkpoint': 'checkpoints/',
                     'log': 'logs/'}
USE_LOGGED_WEIGHTS = False
DEFAULT_VALIDATION_SPLIT = './data/validation_split_v1.pkl'


@click.command()
@click.option('--config-name', default='test',
    help='Configuration file to use for model')
class Trainer:
    """Train a translation model."""

    def __init__(self, config_name):
        self.loader = loader

        self.setup_model(config_name)
        self.setup_reload_path()
        self.setup_loader()
        self.setup_chunk_generator()

        self.train()

    def setup_reload_path(self):
        self.named_checkpoint_path = self.named_log_path = self.checkpoint_file_path = None
        if self.name:
            USE_LOGGED_WEIGHTS = True

            local_folder_path           = os.path.join(SAVER_FOLDER_PATH['base'], self.name)
            self.named_checkpoint_path  = os.path.join(local_folder_path, SAVER_FOLDER_PATH['checkpoint'])
            self.named_log_path         = os.path.join(local_folder_path, SAVER_FOLDER_PATH['log'])

            self.checkpoint_file_path   = os.path.join(self.named_checkpoint_path, 'checkpoint')

            # make sure checkpoint folder exists
            if not os.path.exists(self.named_checkpoint_path):
                os.makedirs(self.named_checkpoint_path)
            if not os.path.exists(self.named_log_path):
                os.makedirs(self.named_log_path)

            print("Will read and write from '%s' (checkpoints and logs)" % (local_folder_path))
            if not self.save_freq:
                warn("'save_freq' is 0, won't save checkpoints", UserWarning)

    def setup_placeholders(self):
        # hacked shapes
        self.Xs       = tf.placeholder(tf.float32,   shape=[None, 96, 96, 1], name='X_input')
        self.ts       = tf.placeholder(tf.int32,   shape=[None], name='t_input')

#    def setup_validation_summaries(self):
        """A hack for recording performance metrics with TensorBoard."""

#        valid_summaries = [
#            tf.scalar_summary('validation/accuracy', self.model.accuracy),
#        ]

#        return tf.merge_summary(valid_summaries)

    def setup_model(self, config_name):
        # load the config module to use
        config_path = 'configurations.' + config_name
        config = importlib.import_module(config_path)

        # copy settings that affect the training script
        self.chunk_size = config.Model.chunk_size
        self.name = config.Model.name
        self.log_freq = config.Model.log_freq
        self.save_freq = config.Model.save_freq
        self.valid_freq = config.Model.valid_freq
        self.iterations = config.Model.iterations
        self.tb_log_freq = config.Model.tb_log_freq

        # Create placeholders and construct model
        self.setup_placeholders()
        self.model = config.Model(
                Xs=self.Xs,
                ts=self.ts)

    def setup_loader(self):
        self.sample_generator = dict()
        self.load_method = dict()
        print('making loader ...')
        self.load_method['train'] = loader.LoadMethod(
            loader.paths['train'],
            loader.labels_train)
        print('loading split ...')
        
        split = pickle.load(open(DEFAULT_VALIDATION_SPLIT, 'br'))
        print('making sample gen ...')
        self.sample_generator['train'] = loader.SampleGenerator(
                    self.load_method['train'],
                    permutation=split['indices_train'],
                    shuffle=True,
                    repeat=True)

        # data loader for eval
        # notice repeat = false
        self.eval_sample_generator = {
            'train': loader.SampleGenerator(
                self.load_method['train'],
                permutation=split['indices_train']),
            'validation': loader.SampleGenerator(
                self.load_method['train'],  # TODO: is this the correct load method?
                permutation=split['indices_valid'])}

    def setup_chunk_generator(self):
        print('making chunk gen ...')
        self.chunk_generator = dict()
        self.chunk_generator['train'] = loader.ChunkGenerator(
            self.sample_generator['train'],
            chunk_size=self.chunk_size,
            labels=True)

        # If we have a validation frequency
        # setup needed evaluation generators
        if self.valid_freq:
            self.eval_chunk_generator = {
                'train': loader.ChunkGenerator(
                    self.eval_sample_generator['train'],
                    chunk_size = self.chunk_size,
                    labels = True),
                'validation': loader.ChunkGenerator(
                    self.eval_sample_generator['validation'],
                    chunk_size=self.chunk_size,
                    labels = True)}

    def train(self):
        print("## INITIATE TRAIN")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver = tf.train.Saver()
            # restore only if files exist
            if USE_LOGGED_WEIGHTS and os.path.exists(self.named_checkpoint_path) and os.listdir(self.named_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(self.named_checkpoint_path)
                saver.restore(sess, latest_checkpoint)
            else:
                tf.initialize_all_variables().run()

            # prepare summary operations and summary writer
#            summaries = tf.merge_all_summaries()
#            self.val_summaries = self.setup_validation_summaries()
#            if self.named_log_path and os.path.exists(self.named_log_path):
#                writer = tf.train.SummaryWriter(self.named_log_path, sess.graph_def)
#                self.writer = writer

            print("## TRAINING...")
            combined_time = 0.0  # total time for each print
            for i, t_chunk in enumerate(self.chunk_generator['train'].gen_chunk()):
                if self.valid_freq and i % self.valid_freq == 0:
                    self.validate(sess)

                ## TRAIN START ##
                fetches = [
                        self.model.loss,
#                        self.model.ys,
#                        summaries,
                        self.model.train_op,
                        self.model.accuracy ]

                res, elapsed_it = self.perform_iteration(
                    sess,
                    fetches,
                    chunk=t_chunk)
                ## TRAIN END ##

                combined_time += elapsed_it

#                if self.named_log_path and os.path.exists(self.named_log_path) and i % self.tb_log_freq == 0:
#                    writer.add_summary(res[2], i)

                if self.save_freq and i and i % self.save_freq == 0 and self.named_checkpoint_path:
                    saver.save(sess, self.checkpoint_file_path, self.model.global_step)

                if self.log_freq and i % self.log_freq == 0:
                    chunk_acc = res[2]
                    click.echo('Iteration %i\t Loss: %f\t Acc: %f\t Elapsed: %f (%f)' % (
                        i, np.mean(res[0]), chunk_acc, combined_time, (combined_time/self.log_freq) ))
                    combined_time = 0.0

#                if i >= self.iterations:
#                    click.echo('reached max iteration: %d' % i)
#                    break

    def perform_iteration(self, sess, fetches, feed_dict=None, chunk=None):
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
                    self.Xs:     chunk['X'],
                    self.ts:     chunk['t']}

        start_time = time.time()
        res = sess.run(fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

        return (res, elapsed)

    def validate(self, sess):
        print("## VALIDATING")
        accuracies = []
        valid_ys = []
        valid_ts = []
        for v_chunk in self.eval_chunk_generator['validation'].gen_chunk():
            fetches = [self.model.accuracy]

            res, time = self.perform_iteration(
                sess,
                fetches,
                feed_dict=None,
                chunk=v_chunk)

            # TODO: accuracies should be weighted by batch sizes
            # before averaging
#            valid_ys.append(res[1])
#            valid_ts.append(v_chunk['t_encoded'])
            accuracies.append(res[0])

#        valid_ys = np.concatenate(valid_ys, axis=0)
#        valid_ts = np.concatenate(valid_ts, axis=0)

        # convert all predictions to strings
#        str_ts, str_ys = utils.numpy_to_words(valid_ts,
#                                              valid_ys,
#                                              self.alphabet)

        # accuracy
        valid_acc = np.mean(accuracies)
        print('\t%s%.2f%%' % ('accuracy:'.ljust(25), (valid_acc * 100)))

#        if self.named_log_path and os.path.exists(self.named_log_path):
#            feed_dict = {
#                self.model.accuracy: valid_acc,
#            }
#            fetches = [self.val_summaries, self.model.global_step]
#            summaries, i = sess.run(fetches, feed_dict)
#            self.writer.add_summary(summaries, i)

#        print("\n## VALIDATION DONE")

if __name__ == '__main__':
    trainer = Trainer()
