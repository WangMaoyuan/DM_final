# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import resnet_model
from official.resnet import resnet_run_loop

_LENGTH = 6812
_NUM_CHANNELS = 3
_DEFAULT_BYTES = _LENGTH
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_BYTES + 1
_NUM_CLASSES = 397
_NUM_DATA_FILES = 1

_NUM_IMAGES = {
    'train': 60992,
    'validation': 15248,
}


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""

  assert os.path.exists(data_dir), (
      'check directory existance first')

  if is_training:
    return [os.path.join(data_dir, 'train.record')]
  else:
    return [os.path.join(data_dir, 'test.record')]


def parse_record(raw_record, is_training):
    feature = {}
    feature["label"] = tf.FixedLenFeature([], dtype=tf.int64)
    fs = list(map(lambda x:'feature'+str(x),range(1,6813)))
    for i in range(len(_LENGTH)):
        feature[fs[i]] = tf.FixedLenFeature([], dtype=tf.float32)
    features = tf.parse_single_example(raw_record, feature)

    imageFeature = [features['feature%d' % (i + 1)] for i in range(_LENGTH)]
    label = tf.one_hot(features['label'], _NUM_CLASSES)
    return imageFeature, label



def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required
      currently to handle the batch leftovers, and can be removed
      when that is handled directly by Estimator.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['validation']

  return resnet_run_loop.process_record_dataset(
      dataset, is_training, batch_size, _NUM_IMAGES['train'],
      parse_record, num_epochs, num_parallel_calls,
      examples_per_epoch=num_images, multi_gpu=multi_gpu)


# def get_synth_input_fn():
#   return resnet_run_loop.get_synth_input_fn(
#       _LENGTH, _NUM_CHANNELS, _NUM_CLASSES)
# check shape of run_loop.get_synth_input_fn

###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               version=resnet_model.DEFAULT_VERSION):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        version=version,
        data_format=data_format)


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
      decay_rates=[1, 0.1, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(features, labels, mode, Cifar10Model,
                                         resnet_size=params['resnet_size'],
                                         weight_decay=weight_decay,
                                         learning_rate_fn=learning_rate_fn,
                                         momentum=0.9,
                                         data_format=params['data_format'],
                                         version=params['version'],
                                         loss_filter_fn=loss_filter_fn,
                                         multi_gpu=params['multi_gpu'])


def main(argv):
  parser = resnet_run_loop.ResnetArgParser()
  # Set defaults that are reasonable for this model.
  parser.set_defaults(data_dir='./data',
                      model_dir='./res_model',
                      resnet_size=32,
                      train_epochs=250,
                      epochs_between_evals=10,
                      batch_size=128)

  flags = parser.parse_args(args=argv[1:])

  input_function = flags.use_synthetic_data and get_synth_input_fn() or input_fn

  resnet_run_loop.resnet_main(
      flags, cifar10_model_fn, input_function,
      shape=[_LENGTH, _NUM_CHANNELS])
# check shape of resnet_main

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
