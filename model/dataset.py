import os
import sys

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
import davis
import logging
import skimage
from utils.util import convert_type
from scipy.misc import imresize
from sys import version_info

from functools import partial


logging.basicConfig(level=logging.INFO)


def expand_image(osvos_image):
  # TODO notice b/c FLAGS cannot pass in, we hard code num_classes as 10
  # given a 480*854 image, return a 480*854*num_classes stack of 0, 1
  num_object = 10
  osvos_expand = np.zeros((480, 854, num_object))
  for layer in range(num_object):
    tmp = np.zeros_like(osvos_image)
    tmp[osvos_image == layer] = 1
    osvos_expand[:, :, layer] = tmp
  return osvos_expand


def load_osvos(osvos_file):
  osvos_image, _ = davis.io.imread_indexed(osvos_file)
  if osvos_image.shape != (480, 854):
    raise Exception("Invalid dimension {} from osvos path {}, resize".format(osvos_image.shape, osvos_file))
    # osvos_image = imresize(osvos_image, (480, 854, 1))
  return expand_image(osvos_image).astype(np.float32)

def load_maskrcnn(maskrcnn_file):
  maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_file)
  maskrcnn_image = maskrcnn_image[..., np.newaxis]
  if maskrcnn_image.shape != (480, 854, 1):
    raise Exception("Invalid dimension {} from markrcnn path {}, resize".format(maskrcnn_image.shape, maskrcnn_file))
    # maskrcnn_image = imresize(maskrcnn_image, (480, 854, 1))
  return maskrcnn_image.astype(np.float32)

def load_jpg(groundtruth_image_file):
  groundtruth_image = skimage.io.imread(groundtruth_image_file)
  if groundtruth_image.shape != (480, 854, 3):
    raise Exception(
      "Invalid dimension {} from groundtruth path {}, resize".format(groundtruth_image.shape, groundtruth_image_file))
    # groundtruth_image = imresize(groundtruth_image, (480, 854, 3))
  return groundtruth_image.astype(np.float32)


def load_firstframe(firstframe_image_file):
  firstframe_image, _ = davis.io.imread_indexed(firstframe_image_file)
  if firstframe_image.shape != (480, 854):
    raise Exception(
      "Invalid dimension {} from firstframe path {}, resize".format(firstframe_image.shape, firstframe_image_file))
  return expand_image(firstframe_image).astype(np.float32)


def load_groundtruth_label(groundtruth_label_file):
  groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_file)
  if groundtruth_label_image.shape != (480, 854):
    logging.warn(
      "Invalid dimension {} from path {}, resize".format(groundtruth_label_image.shape, groundtruth_label_file))
    # groundtruth_image = imresize(groundtruth_image, (480, 854, 1))
    raise Exception("Invalid dimension {}".format(groundtruth_label_image.shape))
  return expand_image(groundtruth_label_image).astype(np.float32)


def _read_py_function_134(osvos_file,
                                 maskrcnn_file,
                                 groundtruth_label_file,
                                 groundtruth_image_file,
                                 firstframe_image_file):
  osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file = \
    convert_type(osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file)

  input_images = []
  input_images.append(load_osvos(osvos_file))
  input_images.append(load_jpg(groundtruth_image_file))
  input_images.append(load_firstframe(firstframe_image_file))
  input = np.concatenate(tuple(input_images), axis=2)
  groundtruth_label_image = load_groundtruth_label(groundtruth_label_file)

  logging.debug("################### input shape {} type {} dtype {}".format(input.shape, type(input), input.dtype))
  logging.debug("################### groundtruth_label_image shape {} dtype {}".format(groundtruth_label_image.shape,
                                                                                      groundtruth_label_image.dtype))
  return input, groundtruth_label_image


def _read_py_function_12(osvos_file,
                         maskrcnn_file,
                         groundtruth_label_file,
                         groundtruth_image_file,
                         firstframe_image_file):
  osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file = \
    convert_type(osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file)

  input_images = []
  input_images.append(load_osvos(osvos_file))
  input_images.append(load_maskrcnn(maskrcnn_file))

  input = np.concatenate(tuple(input_images), axis=2)
  groundtruth_label_image = load_groundtruth_label(groundtruth_label_file)

  return input, groundtruth_label_image


def _read_py_function_34(osvos_file,
                         maskrcnn_file,
                         groundtruth_label_file,
                         groundtruth_image_file,
                         firstframe_image_file):

  osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file = \
    convert_type(osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file)

  input_images = []
  input_images.append(load_jpg(groundtruth_image_file))
  input_images.append(load_firstframe(firstframe_image_file))

  input = np.concatenate(tuple(input_images), axis=2)
  groundtruth_label_image = load_groundtruth_label(groundtruth_label_file)
  return input, groundtruth_label_image


def _read_py_function_1234(osvos_file,
                           maskrcnn_file,
                           groundtruth_label_file,
                           groundtruth_image_file,
                           firstframe_image_file):
  osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file = \
    convert_type(osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file)

  input_images = []
  input_images.append(load_osvos(osvos_file))
  input_images.append(load_maskrcnn(maskrcnn_file))
  input_images.append(load_jpg(groundtruth_image_file))
  input_images.append(load_firstframe(firstframe_image_file))

  input = np.concatenate(tuple(input_images), axis=2)
  groundtruth_label_image = load_groundtruth_label(groundtruth_label_file)
  return input, groundtruth_label_image


def load_data(FLAGS, osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths,
              firstframe_image_paths):
  file_tuple = (osvos_files, maskrcnn_files, groundtruth_label_files, groundtruth_image_paths, firstframe_image_paths)

  training_dataset = tf.data.Dataset.from_tensor_slices(file_tuple)
  python_function = None

  if FLAGS.enable_osvos and FLAGS.enable_maskrcnn and not FLAGS.enable_jpg and not FLAGS.enable_firstframe:
    python_function = _read_py_function_12
  elif not FLAGS.enable_osvos and not FLAGS.enable_maskrcnn and FLAGS.enable_jpg and FLAGS.enable_firstframe:
    python_function = _read_py_function_34
  elif FLAGS.enable_osvos and FLAGS.enable_maskrcnn and FLAGS.enable_jpg and FLAGS.enable_firstframe:
    python_function = _read_py_function_1234
  elif FLAGS.enable_osvos and FLAGS.enable_firstframe and not FLAGS.enable_maskrcnn and FLAGS.enable_jpg:
    python_function = _read_py_function_134
  else:
    python_function = None
    raise Exception("Not a valid model combination")

  logging.info("python_funtion is {}".format(python_function.__name__))
  logging.info("cv2 version {}".format(cv2.__version__))
  logging.info("skimage version {}".format(skimage.__version__))
  training_dataset = training_dataset.map(
    lambda osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file: tuple(
      tf.py_func(python_function,
                 [osvos_file, maskrcnn_file, groundtruth_label_file, groundtruth_image_file, firstframe_image_file],
                 [tf.float32, tf.float32])
    )
  )
  sequence_list = [file.split('/')[-2] for file in osvos_files]
  logging.debug("Dataset type{},  shape {}, classes {}".format(
    training_dataset.output_types,
    training_dataset.output_shapes,
    training_dataset.output_classes))
  return training_dataset, sequence_list


def get_channel_dim(FLAGS):
  dim = 0
  if (FLAGS.enable_maskrcnn):
    dim += 1
  if (FLAGS.enable_osvos):
    dim += 10
  if (FLAGS.enable_jpg):
    dim += 3
  if (FLAGS.enable_firstframe):
    dim += 10
  return dim


def dimension_validation(
        osvos_files,
        maskrcnn_files,
        groundtruth_label_files,
        groundtruth_image_files,
        firstframe_image_files,
        logger):
  logger.debug("In check dimension")
  all_image_found = True
  invalid_seqs = set()

  for index in range(len(osvos_files)):
    try:
      osvos_image, _ = davis.io.imread_indexed(osvos_files[index])
      maskrcnn_image, _ = davis.io.imread_indexed(maskrcnn_files[index])
      groundtruth_label_image, _ = davis.io.imread_indexed(groundtruth_label_files[index])
      groundtruth_image = skimage.io.imread(groundtruth_image_files[index])
      firstframe_image, _ = davis.io.imread_indexed(firstframe_image_files[index])

      osvos_image = osvos_image[..., np.newaxis]
      maskrcnn_image = maskrcnn_image[..., np.newaxis]
      firstframe_image = firstframe_image[..., np.newaxis]

      if osvos_image.shape != (480, 854, 1):
        logger.info("Invalid dimension {} from osvos path {}".format(osvos_image.shape, osvos_files[index]))
        raise Exception("Invalid dimension {} from osvos path {}".format(osvos_image.shape, osvos_files[index]))

      if maskrcnn_image.shape != (480, 854, 1):
        logger.info("Invalid dimension {} from maskrcnn path {}".format(maskrcnn_image.shape, maskrcnn_files[index]))
        invalid_seqs.add(osvos_files[index].split('/')[-2])
        raise Exception(
          "Invalid dimension {} from maskrcnn path {}".format(maskrcnn_image.shape, maskrcnn_image[index]))

      if groundtruth_label_image.shape != (480, 854):
        logger.info(
          "Invalid dimension {} from path {}".format(groundtruth_label_image.shape, groundtruth_label_files[index]))
        raise Exception("Invalid dimension {} from label path {}".format(groundtruth_label_image.shape,
                                                                         groundtruth_label_files[index]))

      if groundtruth_image.shape != (480, 854, 3):
        logger.info(
          "Invalid dimension {} from path {}".format(groundtruth_image.shape, groundtruth_image_files[index]))
        raise Exception(
          "Invalid dimension {} from label path {}".format(groundtruth_image.shape, groundtruth_image_files[index]))

      if firstframe_image.shape != (480, 854, 1):
        logger.info("Invalid dimension {} from path {}".format(firstframe_image.shape, firstframe_image_files[index]))
        raise Exception(
          "Invalid dimension {} from label path {}".format(firstframe_image.shape, firstframe_image_files[index]))

      if np.max(groundtruth_label_image) >= 10:
        logger.info(
          " wrong # of numclasses {} from path {}".format(np.max(groundtruth_label_image),
                                                          groundtruth_label_files[index]))
        raise Exception("wrong numclasses {} from label path {}".format(np.max(groundtruth_label_image),
                                                                        groundtruth_label_files[index]))


    except Exception as e:
      logger.error(e)
      all_image_found = False
      continue

  if all_image_found is False:
    logger.error("Invalid sequence {}".format(invalid_seqs))

  return all_image_found
