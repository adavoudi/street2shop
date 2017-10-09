
""" dataset input module.
"""

import tensorflow as tf


def build_input(data_path, batch_size, width, height, mode):
    """Build images and labels.

    Args:
      batch_size: Input batch size.
    Returns:
      images_left: Batches of images. [batch_size, image_size, image_size, 3]
      images_right: Batches of images. [batch_size, image_size, image_size, 3]
      sim_labels: Batches of similarity labels
      labels_left: Batches of labels. [batch_size, num_classes]
      labels_right: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """

    # get single examples
    image_left, image_right, label_left, label_right, sim_label = \
        read_and_decode_single_example(data_path, width, height, mode)
    # groups examples into batches randomly
    images_left, images_right, labels_left, labels_right, sim_labels = tf.train.shuffle_batch(
        [image_left, image_right, label_left, label_right, sim_label], batch_size=batch_size,
        capacity=2000,
        min_after_dequeue=1000)

    # Display the training images in the visualizer.
    tf.summary.image('images', images_left)
    return images_left, images_right, labels_left, labels_right, sim_labels


def read_and_decode_single_example(data_path, width, height, mode):
    # first construct a queue containing a list of file names.
    # this lets a user split up there dataset in multiple files to keep
    # size down

    data_files = tf.gfile.Glob(data_path)

    filename_queue = tf.train.string_input_producer(data_files,
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label_0': tf.FixedLenFeature([303], tf.int64),
            'label_1': tf.FixedLenFeature([303], tf.int64),
            'simlabel': tf.FixedLenFeature([1], tf.int64),
            'image_0': tf.FixedLenFeature([], tf.string),
            'image_1': tf.FixedLenFeature([], tf.string)
        })

    image_0 = tf.image.decode_jpeg(features['image_0'], channels=3)
    # Reshape image data into the original shape
    image_0 = tf.reshape(image_0, [height, width, 3])

    image_1 = tf.image.decode_jpeg(features['image_1'], channels=3)
    # Reshape image data into the original shape
    image_1 = tf.reshape(image_1, [height, width, 3])

    if mode == 'train':
        rnd_hue = tf.random_uniform([1], -0.3, 0.3)
        image_0 = tf.image.adjust_hue(image_0, rnd_hue)
        image_1 = tf.image.adjust_hue(image_1, rnd_hue)
        rnd_angle_0 = tf.random_uniform([1], -0.2, 0.2)
        image_0 = tf.contrib.image.rotate(image_0, rnd_angle_0)
        rnd_angle_1 = tf.random_uniform([1], -0.2, 0.2)
        image_1 = tf.contrib.image.rotate(image_1, rnd_angle_1)

    # Convert to float32 and scale to [0, 1)
    image_0 = tf.image.convert_image_dtype(image_0, dtype=tf.float32)
    # Finally, rescale to [-1,1] instead of [0, 1)
    image_0 = tf.subtract(image_0, 0.5)
    image_0 = tf.multiply(image_0, 2.0)

    # Convert to float32 and scale to [0, 1)
    image_1 = tf.image.convert_image_dtype(image_1, dtype=tf.float32)
    # Finally, rescale to [-1,1] instead of [0, 1)
    image_1 = tf.subtract(image_1, 0.5)
    image_1 = tf.multiply(image_1, 2.0)

    label_0 = tf.cast(features['label_0'], tf.float32)
    label_0 = (label_0 + 1) / 2
    label_1 = tf.cast(features['label_1'], tf.float32)
    label_1 = (label_1 + 1) / 2

    sim_label = tf.cast(features['simlabel'], tf.float32)

    return image_0, image_1, label_0, label_1, sim_label
