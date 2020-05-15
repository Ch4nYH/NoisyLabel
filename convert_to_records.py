# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_symmetric_random_labels(labels, seed = 1, num_classes = 10, percent = 0.5):

    noisy_labels = np.zeros_like(labels)

    for i in range(num_classes):
        np.random.seed(seed + i)
        all_length = np.sum(labels == i)
        noisy_length = int(num_classes * percent / (num_classes - 1) * all_length)
        new_label = [i] * (all_length - noisy_length) + list(np.random.randint(0, 9, size = noisy_length))
        np.random.seed(seed + i)
        noisy_labels[labels == i] = np.random.permutation(np.array(new_label))

    return noisy_labels
  
def convert_to(data_set, name, noisy = False):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  
  # Begin create noisy label
  
  if noisy:
    labels = make_symmetric_random_labels(labels)
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(labels[index])),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())


def main(unused_argv):
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train', noisy = True)
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)