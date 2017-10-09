# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

import numpy as np

# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit

          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net,
                                depth=unit_depth,
                                depth_bottleneck=unit_depth_bottleneck,
                                stride=1,
                                rate=rate)
            rate *= unit_stride

          else:
            net = block.unit_fn(net,
                                depth=unit_depth,
                                depth_bottleneck=unit_depth_bottleneck,
                                stride=unit_stride,
                                rate=1)
            current_stride *= unit_stride
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


# TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
def batch_norm(name, x, mode, extra_train_ops):
    """Batch normalization."""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

        if mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
            extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            # tf.summary.histogram(mean.op.name, mean)
            # tf.summary.histogram(variance.op.name, variance)
        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y, extra_train_ops


def stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]


def residual(x, in_filter, out_filter, stride, mode, extra_train_ops, relu_leakiness,
              activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x, extra_train_ops = batch_norm('init_bn', x, mode, extra_train_ops)
            x = relu(x, relu_leakiness)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x, extra_train_ops = batch_norm('init_bn', x, mode, extra_train_ops)
            x = relu(x, relu_leakiness)

    with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x, extra_train_ops = batch_norm('bn2', x, mode, extra_train_ops)
        x = relu(x, relu_leakiness)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        # if in_filter != out_filter:
        #     orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        #     orig_x = tf.pad(
        #         orig_x, [[0, 0], [0, 0], [0, 0],
        #                  [(out_filter - in_filter) // 2,
        #                   (out_filter - in_filter) // 2]])
        # x += orig_x
        if stride != [1, 1, 1, 1]:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'SAME')
        if in_filter != out_filter:
            orig_x = conv('pad', orig_x, 1, in_filter, out_filter, [1,1,1,1])

        x += orig_x

    # tf.logging.info('image after unit %s', x.get_shape())
    return x, extra_train_ops


def bottleneck_residual(x, in_filter, out_filter, stride, mode, extra_train_ops, relu_leakiness,
                         activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            x, extra_train_ops = batch_norm('init_bn', x, mode, extra_train_ops)
            x = relu(x, relu_leakiness)
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            orig_x = x
            x, extra_train_ops = batch_norm('init_bn', x, mode, extra_train_ops)
            x = relu(x, relu_leakiness)

    with tf.variable_scope('sub1'):
        x = conv('conv1', x, 1, in_filter, out_filter / 4, stride)

    with tf.variable_scope('sub2'):
        x, extra_train_ops = batch_norm('bn2', x, mode, extra_train_ops)
        x = relu(x, relu_leakiness)
        x = conv('conv2', x, 3, out_filter / 4, out_filter / 4,
                       [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
        x, extra_train_ops = batch_norm('bn3', x, mode, extra_train_ops)
        x = relu(x, relu_leakiness)
        x = conv('conv3', x, 1, out_filter / 4, out_filter,
                       [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = conv('project', orig_x, 1, in_filter,
                                out_filter, stride)
        x += orig_x

    return x, extra_train_ops


def conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    # return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    return tf.nn.relu(x)


def fully_connected(x, out_dim, batch_size):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)


def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])