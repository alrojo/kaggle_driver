import tensorflow as tf
import numpy as np


class Model(object):
    # settings that affect train.py
    chunk_size = 64
    name = None  # (string) For saving logs and checkpoints. (None to disable.)
    visualize_freq = 1000  # Visualize training X, y, and t. (0 to disable.)
    log_freq = 1  # How often to print updates during training.
    save_freq = 100  # How often to save checkpoints. (0 to disable.)
    valid_freq = 50#100  # How often to validate.
    iterations = 32000  # How many iterations to train for before stopping.
    tb_log_freq = 100  # How often to save logs for TensorBoard

    # settings that are local to the model
    num_classes = 10
    learning_rate = 0.001
#    reg_scale = 0.0001
#    clip_norm = 1

    def __init__(self, Xs, ts):
        self.Xs, self.ts = Xs, ts

        self.build()
        self.build_loss()
#        self.build_prediction()
        self.training()

    def build(self):
        print('building model ...')
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')

        W_conv1 = weight_variable([3, 3, 1, 32]) # (patch_size, i_c, o_c)
        b_conv1 = bias_variable([32]) # (o_c)
        h_conv1 = tf.nn.relu(conv2d(self.Xs, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

#        W_conv2 = weight_variable([3, 3, 16, 32])
#        b_conv2 = bias_variable([32])

#        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#        h_pool2 = max_pool_2x2(h_conv2)

#        W_conv3 = weight_variable([3, 3, 32, 64])
#        b_conv3 = bias_variable([64])

#        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#        h_pool3 = max_pool_2x2(h_conv3)

        W_fc1 = weight_variable([48*48*32, 256])
        b_fc1 = bias_variable([256])

        l_reshape = tf.reshape(h_pool1, [-1, 48*48*32])
        h_fc1 = tf.nn.relu(tf.matmul(l_reshape, W_fc1) + b_fc1)

#        keep_prob = tf.placeholder(tf.float32)
#        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([256, 10])
        b_fc2 = bias_variable([10])

        logits = tf.matmul(h_fc1, W_fc2) + b_fc2

        # for debugging network (should write this outside of build)
        self.out_tensor = logits
        # add TensorBoard summaries for all variables
#        tf.contrib.layers.summarize_variables()

    def build_loss(self):
        """Build a loss function and accuracy for the model."""
        print('Building model loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax_y = tf.cast(tf.argmax(self.out_tensor, 1), tf.int64)
            correct = tf.to_float(tf.equal(argmax_y, tf.cast(self.ts, tf.int64)))
            self.accuracy = tf.reduce_mean(correct)

#            tf.scalar_summary('accuracy', self.accuracy)

        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.out_tensor, tf.cast(self.ts, tf.int64), name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            loss = cross_entropy_mean
            #with tf.variable_scope('regularization'):
            #    regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
            #    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            #    reg_term = sum([regularize(param) for param in params])

            #loss = loss + reg_term

#            tf.scalar_summary('loss', loss)

        self.loss = loss

#    def build_prediction(self):
#        with tf.variable_scope('prediction'):
#            self.ys = tf.argmax(self.out_tensor, dimension=1)

    def training(self):
        print('Building model training')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Do gradient clipping
        # NOTE: this is the correct, but slower clipping by global norm.
        # Maybe it's worth trying the faster tf.clip_by_norm()
        # (See the documentation for tf.clip_by_global_norm() for more info)
        #grads_and_vars = optimizer.compute_gradients(self.loss)
        #grads, variables = zip(*grads_and_vars)  # unzip list of tuples
        #clipped_grads, global_norm = tf.clip_by_global_norm(grads,
        #                                                    self.clip_norm)
        #clipped_grads_and_vars = zip(clipped_grads, variables)

        # Create TensorBoard scalar summary for global gradient norm
        #tf.scalar_summary('global gradient norm', global_norm)

        # Create TensorBoard summaries for gradients
        #for grad, var in grads_and_vars:
        #    # Sparse tensor updates can't be summarized, so avoid doing that:
        #    if isinstance(grad, tf.Tensor):
        #        tf.histogram_summary('grad_' + var.name, grad)

        # make training op for applying the gradients
        #self.train_op = optimizer.apply_gradients(clipped_grads_and_vars,
        #                                          global_step=self.global_step)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
