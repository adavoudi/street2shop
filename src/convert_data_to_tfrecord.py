"""Converts images to TFRecords file format with Example protos.

Each record within the TFRecord file is a serialized Example proto.
The Example proto contains the following fields:

  left_image: string containing image in RGB colorspace
  right_image: string containing image in RGB colorspace
  sim_label: integer showing the similarity (0: not similar, 1: similar).
  left_label: integer list of labels for the left image
  right_label: integer list of labels for the right image

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
import threading
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import random
from tqdm import tqdm

tf.app.flags.DEFINE_string('base_img_directory', '/home/alireza/Downloads/consumer-to-shop/Img',
                           'The data directory')
tf.app.flags.DEFINE_string('dataset', 'train', 'the dataset name')
tf.app.flags.DEFINE_string('list_eval_partition',
                           '/home/alireza/Downloads/consumer-to-shop/Eval/list_eval_partition.txt',
                           'Path to the `list_eval_partition.txt` file')
tf.app.flags.DEFINE_string('list_attr_items',
                           '/home/alireza/Downloads/consumer-to-shop/Anno/list_attr_items.txt',
                           'Path to the `list_attr_items.txt` file')
tf.app.flags.DEFINE_string('output_directory', '../tmp/tf-records/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_shards', 10,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 10,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('image_width', 300,
                            'resize the input image to this width')
tf.app.flags.DEFINE_integer('image_height', 300,
                            'resize the input image to this height')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_0, image_1, labels, simlabel):
    """Build an Example proto for an example.

    Args:
      image_0: the 3 channels image of the first image
      image_1: the 3 channels image of the second image
      labels: a tuple of two integer list
      simlabel:  an integer; 0 or 1
    Returns:
      Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'label_0': _int64_feature(labels[0]),
        'label_1': _int64_feature(labels[1]),
        'simlabel': _int64_feature(simlabel),
        'image_0': _bytes_feature(tf.compat.as_bytes(image_0)),
        'image_1': _bytes_feature(tf.compat.as_bytes(image_1))
    }))
    return example


def _resize_keep_aspect_ratio(image, dst_size):

    inp_h = np.float32(image.shape[0])
    inp_w = np.float32(image.shape[1])

    out_hs = int(np.float32(dst_size[0]))
    out_ws = int(np.float32(dst_size[1]))

    h1 = int(out_ws * (inp_h / np.float32(inp_w)))
    w2 = int(out_hs * (inp_w / np.float32(inp_h)))
    if h1 <= out_hs:
        output = cv2.resize(image, (out_ws, h1), interpolation=cv2.INTER_CUBIC)
    else:
        output = cv2.resize(image, (w2, out_hs), interpolation=cv2.INTER_CUBIC)

    out_shape = output.shape

    top = int((dst_size[0] - out_shape[0]) / 2)
    left = int((dst_size[1] - out_shape[1]) / 2)
    down = (dst_size[0] - out_shape[0]) - top
    right = (dst_size[1] - out_shape[1]) - left

    output = cv2.copyMakeBorder(
        output, top, down, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

    return output


def _process_image(filename, width, height):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      height: integer, image height in pixels.
      width:  integer, image width in pixels.
    Returns:
      image: string of the 3 channels image.

    """
    image = cv2.imread(filename)
    if image is None:
        raise("Error: " + filename)

    image = _resize_keep_aspect_ratio(image, (height, width))
    encoded_image = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    return encoded_image[1].tostring()


def _process_image_files_batch(thread_index, ranges, name, filenames,
                               sim_labels, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of tuple of strings;
      sim_labels: list of integers
      labels: list of tuple of integers;
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:

            try:
                image_buffer_0 = _process_image(
                    filenames[i][0], FLAGS.image_width, FLAGS.image_height)
                image_buffer_1 = _process_image(
                    filenames[i][1], FLAGS.image_width, FLAGS.image_height)
            except Exception as e:
                print('SKIPPED: Unexpected error while decoding %s or %s' % filenames[i][0], filenames[i][1])
                continue

            example = _convert_to_example(image_buffer_0, image_buffer_1, labels[i], sim_labels[i])
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 50:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(file_names, sim_labels, labels):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      file_names: list of strings; each string is a path to an image file
      labels: list of integer; each integer identifies the ground truth
    """

    c = list(zip(file_names, sim_labels, labels))
    random.shuffle(c)
    file_names, sim_labels, labels = zip(*c)

    name = FLAGS.dataset
    num_shards = FLAGS.num_shards

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(
        0, len(file_names), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' %
          (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, file_names,
                sim_labels, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(file_names)))
    sys.stdout.flush()


def _generate_negative_pairs(num_pairs):
    """Build a list of all positive pair images.

    Args:
      num_pairs: number of negative pairs to generate

    Returns:
      file_names: list of string tuples; each tuple contains two similar image file path.
      sim_labels: list of integers; all zero.
      labels: list of integer array tuples; each tuple contains two integer array, each for
              one of the images
    """
    print('Generating negative pairs list')

    partition_table = pd.read_table(FLAGS.list_eval_partition, header=None,
                          delim_whitespace=True)

    unique_dirs = list(set([os.path.dirname(item) for item in partition_table[0]]))

    attr_table = pd.read_table(FLAGS.list_attr_items, header=None,
                                    delim_whitespace=True)

    file_names = []
    sim_labels = []
    labels = []

    for row in tqdm(range(0, num_pairs)):
        rnd_indx = [random.randint(0, len(unique_dirs)-1),
                    random.randint(0, len(unique_dirs)-1)]
        dirs = [unique_dirs[T] for T in rnd_indx]
        files = []
        for dir in dirs:
            _dir = os.path.join(FLAGS.base_img_directory, dir)
            _files_in_dir = os.listdir(_dir)
            files.append(os.path.join(_dir, random.choice(_files_in_dir)))

        attrs = [attr[-11:] for attr in dirs]
        attr = attr_table.loc[attr_table[0].isin(attrs)]
        if attr.shape[0] == 2:
            attr = attr.ix[:,1:].values.tolist()
            file_names.append((files[0], files[1]))
            sim_labels.append(0)
            labels.append((attr[0], attr[1]))

    return file_names, sim_labels, labels

def _generate_positive_pairs():
    """Build a list of all positive pair images.

    Returns:
      file_names: list of string tuples; each tuple contains two similar image file path.
      sim_labels: list of integers; all ones.
      labels: list of integer array tuples; each tuple contains two integer array, each for
              one of the images
    """
    print('Generating positive pairs list')

    partition_table = pd.read_table(FLAGS.list_eval_partition, header=None,
                          delim_whitespace=True)

    attr_table = pd.read_table(FLAGS.list_attr_items, header=None,
                                    delim_whitespace=True)

    file_names = []
    sim_labels = []
    labels = []

    for row in tqdm(range(0, partition_table.shape[0])):
        p_id = partition_table.iloc[row, 2]
        attr = attr_table.loc[attr_table[0].isin([p_id])]
        if not attr.empty:
            attr = attr.iloc[0,1:].values.tolist()
            file_names.append((os.path.join(FLAGS.base_img_directory, partition_table.iloc[row, 0]),
                               os.path.join(FLAGS.base_img_directory, partition_table.iloc[row, 1])))
            sim_labels.append(1)
            labels.append((attr, attr))

    return file_names, sim_labels, labels

def main(unused_argv):
    assert not FLAGS.num_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')

    print('Saving results to %s' % FLAGS.output_directory)

    file_names_1, sim_labels_1, labels_1 = _generate_positive_pairs()
    file_names_2, sim_labels_2, labels_2 = _generate_negative_pairs(len(sim_labels_1))
    _process_image_files(file_names_1 + file_names_2, sim_labels_1+sim_labels_2, labels_1+labels_2)


if __name__ == '__main__':
    tf.app.run()
