import os
import re
import sys
import yaml
import numpy as np
import logging
import tensorflow as tf
import davis
import skimage
from sys import version_info

logging.basicConfig(level=logging.DEBUG)


def get_channel_dimension2(FLAGS):
  dim = 0
  if (FLAGS.enable_osvos):
    dim += 1
  if (FLAGS.enable_maskrcnn):
    dim += 1
  if (FLAGS.enable_jpg):
    dim += 3
  if (FLAGS.enable_firstframe):
    dim += 1
  return dim


def get_osvos_layer(FLAGS):
  return 0


def get_jpg_layer(FLAGS):
  dim = 0
  if FLAGS.enable_osvos:
    dim += 1
  if FLAGS.enable_maskrcnn:
    dim += 1
  return dim, dim + 3


def get_firstframe_layer(FLAGS):
  dim = 0
  if FLAGS.enable_osvos:
    dim += 1
  if FLAGS.enable_maskrcnn:
    dim += 1
  if FLAGS.enable_jpg:
    dim += 3
  return dim


def load_image_files(FLAGS, seqs):
  osvos_label_paths = []
  maskrcnn_label_paths = []
  groundtruth_label_paths = []
  groundtruth_image_paths = []
  firstframe_image_paths = []

  for sequence in seqs:
    osvos_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.osvos_label_path, sequence)
    maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.maskrcnn_label_path, sequence)
    groundtruth_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_label_path, sequence)
    groundtruth_image_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_image_path, sequence)
    firstframe_image_path = "{}/{}/{}/00000.png".format(FLAGS.read_path, FLAGS.groundtruth_label_path, sequence)

    for file in os.listdir(groundtruth_label_path):
      if re.match(r'[0-9]+.*\.png', file):
        osvos_label_paths.append(osvos_label_path + file)
        maskrcnn_label_paths.append(maskrcnn_label_path + file)
        groundtruth_label_paths.append(groundtruth_label_path + file)
        groundtruth_image_paths.append(groundtruth_image_path + file.split('.')[0] + ".jpg")
        firstframe_image_paths.append(firstframe_image_path)
        logging.debug(groundtruth_label_path + file)
        logging.debug(osvos_label_path + file)
        logging.debug(maskrcnn_label_path + file)
        logging.debug(groundtruth_image_path + file.split('.')[0] + ".jpg")
        logging.debug(firstframe_image_path)

  logging.debug("Validating file length")
  assert len(osvos_label_paths) == len(groundtruth_label_paths)
  assert len(maskrcnn_label_paths) == len(groundtruth_label_paths)
  assert len(groundtruth_image_paths) == len(groundtruth_label_paths)
  assert len(firstframe_image_paths) == len(groundtruth_label_paths)
  return osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths, groundtruth_image_paths, firstframe_image_paths


def check_image_dimension(FLAGS, logger, seqs):
  for sequence in seqs:
    osvos_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.osvos_label_path, sequence)
    maskrcnn_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.maskrcnn_label_path, sequence)
    groundtruth_label_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_label_path, sequence)
    groundtruth_image_path = "{}/{}/{}/".format(FLAGS.read_path, FLAGS.groundtruth_image_path, sequence)

    osvos_image, _ = davis.io.imread_indexed(osvos_label_path + '00000.png')
    osvos_image = osvos_image[..., np.newaxis]
    logger.debug(
      "osvos_image shape{}, type: {} unique {}".format(osvos_image.shape, type(osvos_image), np.unique(osvos_image)))

    maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_label_path + '00000.png')
    maskrcnn_image = maskrcnn_image[..., np.newaxis]
    logger.debug("maskrcnn_image shape{}, unique {}".format(maskrcnn_image.shape, np.unique(maskrcnn_image)))

    groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_path + '00000.png')
    groundtruth_label_image = groundtruth_label_image[..., np.newaxis]
    logger.debug(
      "groundtruth label shape{}, unique {}".format(groundtruth_label_image.shape, np.unique(groundtruth_label_image)))

    input_image = np.concatenate((osvos_image, maskrcnn_image), axis=2)
    logger.debug("input_image  shape{}, unique {}".format(input_image.shape, np.unique(input_image)))


def load_seq_from_yaml(train_path):
  with open(train_path, 'r') as train_stream:
    train_dict = yaml.load(train_stream)
  train_seqs = train_dict['sequences']
  return train_seqs


def path_config(env):
  if env == "jj":
    tf.app.flags.DEFINE_string("read_path",
                               "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/davis-2017/data/DAVIS",
                               "read_path")
    tf.app.flags.DEFINE_string("output_path",
                               "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/segmentation/Results/",
                               "output_path")
    tf.app.flags.DEFINE_string("config_path",
                               "/Users/jingle.jiang/personal/class/stanford/cs231n/final/video_segmentation/segmentation",
                               "config_path")
    tf.app.flags.DEFINE_string("train_seq_yaml", "train-dev.yaml", "train_seq_yaml")
    tf.app.flags.DEFINE_string("test_seq_yaml", "test-dev.yaml", "test_seq_yaml")
    tf.app.flags.DEFINE_string("train_val_yaml", "train-dev.yaml", "train_val_yaml")
    tf.app.flags.DEFINE_string("test_val_yaml", "test-dev.yaml", "test_val_yaml")
    tf.app.flags.DEFINE_string("device", "/cpu:0", "device")
  elif env == "hyuna915":
    tf.app.flags.DEFINE_string("read_path",
                               "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/davis-2017/data/DAVIS",
                               "read_path")
    tf.app.flags.DEFINE_string("output_path",
                               "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/segmentation/Results/",
                               "output_path")
    tf.app.flags.DEFINE_string("config_path",
                               "/Users/hyuna915/Desktop/2018-CS231N/Final_Project/video_segmentation/segmentation",
                               "config_path")
    tf.app.flags.DEFINE_string("train_seq_yaml", "train-dev.yaml", "train_seq_yaml")
    tf.app.flags.DEFINE_string("test_seq_yaml", "test-dev.yaml", "test_seq_yaml")
    tf.app.flags.DEFINE_string("train_val_yaml", "train-dev.yaml", "train_val_yaml")
    tf.app.flags.DEFINE_string("test_val_yaml", "test-dev.yaml", "test_val_yaml")
    tf.app.flags.DEFINE_string("device", "/cpu:0", "device")

  elif env == "cloud":
    tf.app.flags.DEFINE_string("read_path", "/home/shared/video_segmentation/davis-2017/data/DAVIS", "read_path")
    tf.app.flags.DEFINE_string("output_path", "/home/shared/video_segmentation/segmentation/Results/", "output_path")
    tf.app.flags.DEFINE_string("config_path",
                               "/home/shared/video_segmentation/segmentation",
                               "config_path")
    tf.app.flags.DEFINE_string("train_seq_yaml", "train.yaml", "train_seq_yaml")
    tf.app.flags.DEFINE_string("test_seq_yaml", "test.yaml", "test_seq_yaml")
    tf.app.flags.DEFINE_string("train_val_yaml", "train-sample.yaml", "train_val_yaml")
    tf.app.flags.DEFINE_string("test_val_yaml", "test.yaml", "test_val_yaml")
    tf.app.flags.DEFINE_string("device", "/gpu:0", "device")


def convert_type(osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file):
  logging.debug("Type of osvos_file {}".format(type(osvos_file)))
  logging.debug("Type of maskrcnn_file {}".format(type(maskrcnn_file)))
  logging.debug("Type of groundtruth_image_file {}".format(type(groundtruth_image_file)))
  logging.debug("Type of firstframe_image_file {}".format(type(firstframe_image_file)))

  if version_info[0] > 2:
    osvos_file = osvos_file.decode("utf-8")
    maskrcnn_file = maskrcnn_file.decode("utf-8")
    groundtruth_label_file = groundtruth_label_file.decode("utf-8")
    groundtruth_image_file = groundtruth_image_file.decode("utf-8")
    firstframe_image_file = firstframe_image_file.decode("utf-8")
    logging.debug("Python 3 detected, convert all bytes to string")
    logging.debug("Type of osvos_file {}".format(type(osvos_file)))
    logging.debug("Type of maskrcnn_file {}".format(type(maskrcnn_file)))
    logging.debug("Type of groundtruth_label_file {}".format(type(groundtruth_label_file)))
    logging.debug("Type of groundtruth_image_file {}".format(type(groundtruth_image_file)))
    logging.debug("Type of firstframe_image_file {}".format(type(firstframe_image_file)))

  return osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file


def write_summary(value, tag, summary_writer, global_step):
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=value)
  summary_writer.add_summary(summary, global_step)


def print_image(FLAGS, seq_name_, image_number_, object_number_, x_np, y_np, pred_mask_, epoch):
  for idx in range(len(seq_name_)):
    if (seq_name_[idx].decode("utf-8") == "tennis" and image_number_[idx].decode("utf-8") == "00057.png" and
        object_number_[idx] == 1) or \
            (seq_name_[idx].decode("utf-8") == "swing" and image_number_[idx].decode("utf-8") == "00023.png" and
             object_number_[idx] == 1) or \
            (seq_name_[idx].decode("utf-8") == "surf" and image_number_[idx].decode("utf-8") == "00008.png"):
      # we save
      seq_name__ = seq_name_[idx].decode("utf-8")
      image_number__ = image_number_[idx].decode("utf-8")
      obj_number = str(object_number_[idx])
      savedir = os.path.join(FLAGS.output_path, FLAGS.model_label, "visualize", seq_name__, image_number__, "obj_" + obj_number)
      if not os.path.exists(savedir):
        os.makedirs(savedir)

      if FLAGS.enable_osvos and epoch == 0:
        # since osvos does not change, we only save the first epoch
        osvos_layer = get_osvos_layer(FLAGS)
        davis.io.imwrite_indexed(os.path.join(savedir, 'osvos_epoch.png'),
                                 x_np[idx, :, :, osvos_layer].astype('uint8'))

      if FLAGS.enable_firstframe and epoch == 0:
        firstframe_layer = get_firstframe_layer(FLAGS)
        davis.io.imwrite_indexed(os.path.join(savedir, 'firstframe.png'),
                                 x_np[idx, :, :, firstframe_layer].astype('uint8'))

      if FLAGS.enable_jpg and epoch == 0:
        jpg_layer_start, jpg_layer_end = get_jpg_layer(FLAGS)
        skimage.io.imsave(os.path.join(savedir, 'gt_image.jpg'), x_np[idx, :, :, jpg_layer_start:jpg_layer_end])

      if epoch == 0:
        davis.io.imwrite_indexed(os.path.join(savedir, 'gt_label_epoch.png'),
                                 y_np[idx, :, :, 0].astype('uint8'))

      _pred_to_save = np.zeros((FLAGS.height, FLAGS.weight), dtype='uint8')
      _pred_to_save[pred_mask_[idx, :, :, 0] > 0.5] = 1
      davis.io.imwrite_indexed(os.path.join(savedir, 'predict_epoch{}.png'.format(str(epoch))), _pred_to_save)
