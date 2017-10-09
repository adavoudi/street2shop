import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple
import resnet_v1
import pyramid_network
import resnet_utils
import utils

HParams = namedtuple('HParams',
                     'min_lrn_rate, lrn_rate, weight_decay_rate, optimizer, initial_weights_path')


class Model(object):
    """Implementation of the Model."""

    def __init__(self, hps, mode,
                 images_left=None, images_right=None, labels_left=None, labels_right=None, sim_labels=None):
        """
        Args:
          hps: Hyper parameters.
          mode: One of 'train' or 'eval'.
        """
        self.hps = hps
        self.mode = mode

        if images_left is None:
            self.image_left = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image_left')
            if mode == 'train':
                self.image_right = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image_right')
                self.similarity_label = tf.placeholder(tf.float32, [None, 1], name='label_sim')
                self.image_left_label = tf.placeholder(tf.float32, [None, 50], name='label_left')
                self.image_right_label = tf.placeholder(tf.float32, [None, 50], name='label_right')
        else:
            self.image_left = tf.placeholder_with_default(images_left, [None, 300, 300, 3], name='image_left')
            if mode == 'train':
                self.image_right = tf.placeholder_with_default(images_right, [None, 300, 300, 3], name='image_right')
                self.similarity_label = tf.placeholder_with_default(sim_labels, [None, 1], name='label_sim')
                self.image_left_label = tf.placeholder_with_default(labels_left, [None, 303], name='label_left')
                self.image_right_label = tf.placeholder_with_default(labels_right, [None, 303], name='label_right')

    def build_graph(self):
        """Build a whole graph for the model."""

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_accuracy_cost()
            self._build_train_op()

    def _build_accuracy_cost(self):
        """Build training specific ops for the graph."""

        with tf.variable_scope("loss"):
            self.similarity_loss = self._similarity_loss(self.feature_left, self.feature_right, self.similarity_label,
                                                        0.5, scope='similarity_loss')
            left_cat_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.image_left_label, logits=self.logits_left,
                                                                    name='left_cat_loss')
            right_cat_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.image_right_label, logits=self.logits_right,
                                                                     name='right_cat_loss')

            self.cat_loss = left_cat_loss + right_cat_loss
            self.cost = tf.add(self.similarity_loss, self.cat_loss, name='total_loss')

    def _build_model(self):
        """ build the computational graph of the Model
        """

        with tf.variable_scope("Feature_Extractor") as scope:

            self.feature_left, self.extra_train_ops = self._feature_extractor(self.image_left, 'train', 'Feature_Extractor')
            self.feature_left = tf.layers.dropout(self.feature_left, rate=0.5, training=self.mode == 'train')
            if self.mode == 'train':
                scope.reuse_variables()
                self.feature_right, _ = self._feature_extractor(self.image_right, 'train', 'Feature_Extractor')
                self.feature_right = tf.layers.dropout(self.feature_right, rate=0.5, training=self.mode == 'train')
            else:
                self.feature = self.feature_left

        with tf.variable_scope("cat_logits") as scope:
            num_classes = self.image_left_label.get_shape().as_list()[1]
            self.logits_left = slim.fully_connected(self.feature_left, num_classes, activation_fn=None, scope='logits')
            if self.mode == 'train':
                scope.reuse_variables()
                self.logits_right = slim.fully_connected(self.feature_right, num_classes, activation_fn=None, scope='logits')
            else:
                self.logits = self.logits_left

    def _feature_extractor(self, input, mode, scope=None, relu_leakiness=0.1):
        image = tf.placeholder_with_default(input, (None, 300, 300, 3), 'input_image')

        pyramid_map = {'C1': 'FeatureX1/resnet_v1_50/conv1/Relu:0',
                       'C2': 'FeatureX1/resnet_v1_50/block1/unit_2/bottleneck_v1',
                       'C3': 'FeatureX1/resnet_v1_50/block2/unit_3/bottleneck_v1',
                       'C4': 'FeatureX1/resnet_v1_50/block3/unit_5/bottleneck_v1',
                       'C5': 'FeatureX1/resnet_v1_50/block4/unit_3/bottleneck_v1',
                       }

        if scope is not None:
            for key, value in pyramid_map.iteritems():
                pyramid_map[key] = scope + "/" + value

        with tf.variable_scope("FeatureX1"):
            with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.000005)):
                logits, end_points = resnet_v1.resnet_v1_50(image, 1000, is_training=self.mode=='train')

            pyramid = pyramid_network.build_pyramid(pyramid_map, end_points)

        extra_train_ops = []
        py_features = [pyramid['P5']]

        with tf.variable_scope("FeatureX2"):
            with tf.variable_scope("pyramid_2"):
                x = pyramid['P2']

                with tf.variable_scope("block_0"):
                    x, extra_train_ops = resnet_utils.residual(x, 256, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=True)

                with tf.variable_scope("block_1"):
                    x, extra_train_ops = resnet_utils.residual(x, 64, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=False)

                with tf.variable_scope("block_2"):
                    x, extra_train_ops = resnet_utils.residual(x, 64, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=False)

                py_features.append(x)

            with tf.variable_scope("pyramid_3"):
                x = pyramid['P3']

                with tf.variable_scope("block_0"):
                    x, extra_train_ops = resnet_utils.residual(x, 256, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=True)

                with tf.variable_scope("block_1"):
                    x, extra_train_ops = resnet_utils.residual(x, 64, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=False)

                py_features.append(x)

            with tf.variable_scope("pyramid_4"):
                x = pyramid['P4']

                with tf.variable_scope("block_0"):
                    x, extra_train_ops = resnet_utils.residual(x, 256, 64,
                                                               resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                               relu_leakiness,
                                                               activate_before_residual=True)

                py_features.append(x)

            x = tf.concat(py_features, axis=3, name='concat')

            with tf.variable_scope("block_0"):
                x, extra_train_ops = resnet_utils.residual(x, 448, 256,
                                                           resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                           relu_leakiness,
                                                           activate_before_residual=True)

            with tf.variable_scope("block_1"):
                x, extra_train_ops = resnet_utils.residual(x, 256, 256,
                                                           resnet_utils.stride_arr(2), mode, extra_train_ops,
                                                           relu_leakiness,
                                                           activate_before_residual=False)

            global_avg = tf.reduce_mean(x, [1, 2], name='global_avg')

        feature = tf.nn.l2_normalize(global_avg, 0, name='Feature')

        return feature, extra_train_ops

    def _build_train_op(self):
        """Build training specific ops for the graph."""

        with tf.variable_scope("optimizer"):
            self.lrn_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, name='learning_rate', trainable=False)
            tf.summary.scalar('learning_rate', self.lrn_rate)
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(self.cost, trainable_variables)
            if self.hps.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate)
            apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step, name='train_step')

            train_ops = [apply_op]
            train_ops = train_ops + self.extra_train_ops

            self.train_op = tf.group(*train_ops)

    def _similarity_loss(self, feature_left, feature_right, label, margin, scope):
        with tf.variable_scope(scope):
            cosine_dist = tf.losses.cosine_distance(feature_left, feature_right, dim=0)

            match_loss = tf.minimum(margin, cosine_dist, 'match_term')
            mismatch_loss = tf.maximum(0., tf.subtract(margin, cosine_dist), 'mismatch_term')

            # if label is 1, only match_loss will count, otherwise mismatch_loss
            loss = tf.add(tf.multiply(label, match_loss), \
                          tf.multiply((1 - label), mismatch_loss), 'loss_add')

            loss_mean = tf.reduce_mean(loss)

        return loss_mean

    def load_initial_weights(self, session):
        """Load weights from file into feature extractor model. """

        featureX1_vars = utils.collect_vars('Feature_Extractor/FeatureX1')
        restorer = tf.train.Saver(var_list=featureX1_vars)
        restorer.restore(session, self.hps.initial_weights_path)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))