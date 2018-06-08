import davis
import logging
import skimage

import numpy as np
import tensorflow as tf
from utils.util import convert_type, get_channel_dimension2

from functools import partial

def gen_134(file_tuple=None):
  assert file_tuple is not None
  (osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths) = file_tuple
  for (osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_path, firstframe_image_path) in \
          zip(osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths):

    seq_name = groundtruth_label_file.split('/')[-2]
    image_num = groundtruth_label_file.split('/')[-1]
    osvos_im, _ = davis.io.imread_indexed(osvos_file)
    # maskrcnn_im, _ = davis.io.imread_indexed(maskrcnn_file)
    groundtruth_label_im, _ = davis.io.imread_indexed(groundtruth_label_file)
    rgb_img = skimage.io.imread(groundtruth_image_path) / 255
    first_img, _ = davis.io.imread_indexed(firstframe_image_path)

    num_classes = np.max(groundtruth_label_im)

    for class_ in range(1, num_classes + 1):
      osvos_per_class = np.zeros_like(osvos_im)
      osvos_per_class[osvos_im == class_] = 1

      gt_per_class = np.zeros_like(groundtruth_label_im)
      gt_per_class[groundtruth_label_im == class_] = 1

      firstimage_per_class = np.zeros_like(first_img)
      firstimage_per_class[first_img == class_] = 1

      input_images = []
      input_images.append(osvos_per_class[..., np.newaxis])
      input_images.append(rgb_img)
      input_images.append(firstimage_per_class[..., np.newaxis])
      input = np.concatenate(tuple(input_images), axis=2)

      yield (seq_name, image_num, class_, input, gt_per_class[..., np.newaxis])


def gen_34(file_tuple=None):
  assert file_tuple is not None
  (osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths) = file_tuple
  for (osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_path, firstframe_image_path) in \
          zip(osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths):

    seq_name = groundtruth_label_file.split('/')[-2]
    image_num = groundtruth_label_file.split('/')[-1]
    groundtruth_label_im, _ = davis.io.imread_indexed(groundtruth_label_file)
    rgb_img = skimage.io.imread(groundtruth_image_path) / 255
    first_img, _ = davis.io.imread_indexed(firstframe_image_path)

    num_classes = np.max(groundtruth_label_im)

    for class_ in range(1, num_classes + 1):
      gt_per_class = np.zeros_like(groundtruth_label_im)
      gt_per_class[groundtruth_label_im == class_] = 1

      firstimage_per_class = np.zeros_like(first_img)
      firstimage_per_class[first_img == class_] = 1

      input_images = []
      input_images.append(rgb_img)
      input_images.append(firstimage_per_class[..., np.newaxis])
      input = np.concatenate(tuple(input_images), axis=2)

      yield (seq_name, image_num, class_, input, gt_per_class[..., np.newaxis])


def gen_1234(file_tuple=None):
  assert file_tuple is not None
  (osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths) = file_tuple
  for (osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_path, firstframe_image_path) in \
          zip(osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths):

    seq_name = groundtruth_label_file.split('/')[-2]
    image_num = groundtruth_label_file.split('/')[-1]
    osvos_im, _ = davis.io.imread_indexed(osvos_file)
    maskrcnn_im, _ = davis.io.imread_indexed(maskrcnn_file)
    groundtruth_label_im, _ = davis.io.imread_indexed(groundtruth_label_file)
    rgb_img = skimage.io.imread(groundtruth_image_path) / 255
    first_img, _ = davis.io.imread_indexed(firstframe_image_path)

    num_classes = np.max(groundtruth_label_im)

    for class_ in range(1, num_classes + 1):
      osvos_per_class = np.zeros_like(osvos_im)
      osvos_per_class[osvos_im == class_] = 1

      gt_per_class = np.zeros_like(groundtruth_label_im)
      gt_per_class[groundtruth_label_im == class_] = 1

      maskrcnn_per_class = np.zeros_like(maskrcnn_im)
      maskrcnn_per_class[maskrcnn_im==class_] = 1

      firstimage_per_class = np.zeros_like(first_img)
      firstimage_per_class[first_img == class_] = 1

      input_images = []
      input_images.append(osvos_per_class[..., np.newaxis])
      input_images.append(maskrcnn_per_class[...,np.newaxis])
      input_images.append(rgb_img)
      input_images.append(firstimage_per_class[..., np.newaxis])
      input = np.concatenate(tuple(input_images), axis=2)

      yield (seq_name, image_num, class_, input, gt_per_class[..., np.newaxis])


def load_data2(FLAGS, osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths,
               firstframe_image_paths):
  file_tuple = (osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths)
  gen_func = None

  if FLAGS.enable_osvos and FLAGS.enable_firstframe and not FLAGS.enable_maskrcnn and FLAGS.enable_jpg:
    gen_func = gen_134
  elif not FLAGS.enable_osvos and not FLAGS.enable_maskrcnn and FLAGS.enable_jpg and FLAGS.enable_firstframe:
    gen_func = gen_34
  elif FLAGS.enable_osvos and FLAGS.enable_maskrcnn and FLAGS.enable_jpg and FLAGS.enable_firstframe:
    gen_func = gen_1234
  else:
    raise Exception("Not a valid model combination")
  channel_dim = get_channel_dimension2(FLAGS)
  # output is [seq_name, image_#, object_#, input_H*W, groudtruth_H*W_]
  training_dataset = tf.data.Dataset.from_generator(
    partial(gen_func, file_tuple=file_tuple),
    (tf.string, tf.string, tf.int8, tf.float32, tf.float32),
    (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
     tf.TensorShape([FLAGS.height, FLAGS.weight, channel_dim]), tf.TensorShape([480, 854, 1])),
  )
  return training_dataset, [1, 2, 3]


