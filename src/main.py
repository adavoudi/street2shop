import sys
import tensorflow as tf
from tqdm import tqdm
import logging

import tf_input
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'train, eval.')
tf.app.flags.DEFINE_integer('image_width', 300, 'The input image width.')
tf.app.flags.DEFINE_integer('image_height', 300, 'The input image height.')
tf.app.flags.DEFINE_float("init_lrn_rate", 0.00015, 'Initial learning rate')
tf.app.flags.DEFINE_float("min_lrn_rate", 0.00010, 'Min learning rate')
tf.app.flags.DEFINE_string('initial_weights_path', '../tmp/', 'Path to the maskrcnn snapshot')
tf.app.flags.DEFINE_integer('iterations', 30000, 'Number of iterations')
tf.app.flags.DEFINE_string('train_data_path', '../tmp/tf-records/train*',
                           'File pattern for training data.')
tf.app.flags.DEFINE_string('train_dir', '../tmp/train-logs',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_integer('batch_size', 30, 'Size of a batch of images')
tf.app.flags.DEFINE_integer('save_step', 500, 'The save frequency.')
tf.app.flags.DEFINE_string('checkpoint_dir', '../tmp/checkpoints/',
                           'Directory to keep the checkpoints.')

def train(hps):
    """Training loop."""

    images_left, images_right, labels_left, labels_right, sim_labels = \
        tf_input.build_input(FLAGS.train_data_path, FLAGS.batch_size,
                             FLAGS.image_width, FLAGS.image_height, 'train')

    model_train = model.Model(hps, "train", images_left, images_right, labels_left, labels_right, sim_labels)

    model_train.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    config = tf.ConfigProto(allow_soft_placement=True)
    # Create a saver.
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_checkpoint is not None:
            logging.info('Restoring weights from {}:'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        logging.info('Starting queue runners')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        global_step = sess.run(model_train.global_step)

        bar = tqdm(range(global_step, FLAGS.iterations + 1))
        bar.set_description('sim_loss: {:.3f}, cat_loss: {:.3f}, lrn_rate: {:.3f}'.format(100, 100, hps.lrn_rate))
        bar.refresh()

        for step in bar:

            lrn_rate = (1 - (step * 1.0) / FLAGS.iterations) * (hps.lrn_rate - hps.min_lrn_rate) + hps.min_lrn_rate
            sess.run(model_train.lrn_rate.assign(lrn_rate))

            sim_loss, cat_loss, _ = sess.run(
                [model_train.similarity_loss, model_train.cat_loss, model_train.train_op])

            bar.set_description(
                'sim_loss: {:.3f}, cat_loss: {:.3f}, lrn_rate: {:.3f}'.format(sim_loss, cat_loss, lrn_rate))

            if step % FLAGS.save_step == 0:
                snapshot_path = saver.save(sess, FLAGS.checkpoint_dir,
                                           global_step=step)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

def evaluate(hps):
    """Eval."""
    print 'To be implemented'


def main(_):

    hps = model.HParams(min_lrn_rate=FLAGS.min_lrn_rate,
                        lrn_rate=FLAGS.init_lrn_rate,
                        weight_decay_rate=0.0002,
                        optimizer='mom',
                        initial_weights_path=FLAGS.initial_weights_path)

    if FLAGS.mode == 'train':
        train(hps)
    elif FLAGS.mode == 'eval':
        evaluate(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()