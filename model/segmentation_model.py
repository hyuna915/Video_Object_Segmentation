from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

logging.basicConfig(level=logging.INFO)

smoothness = 1.0

def convolution_layer(filters, kernel=(3, 3), activation='relu', input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, padding='same')
  else:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, activation=activation, input_shape=input_shape,
                                  padding='same')


def concatenated_de_convolution_layer(filters, input_shape=None):
  if input_shape is None:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
  else:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                           input_shape=input_shape)


def max_pooling_layer():
  return tf.keras.layers.MaxPooling2D(pool_size=(2, 2))


# This is a uNet similar neural networks structure
def unet_w_connect(FLAGS, channel_dim, inputs):
  logging.info("In unet_w_connect")
  input_shape = (FLAGS.height, FLAGS.weight, channel_dim)
  input_layers=[]
  conv0 = convolution_layer(16)(inputs)
  input_layers.append(conv0)
  conv0 = convolution_layer(16)(conv0)
  input_layers.append(conv0)
  crop0 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,6)))(conv0)
  pool0 = max_pooling_layer()(conv0)
  input_layers.append(pool0)

  conv1 = convolution_layer(32)(pool0)
  input_layers.append(conv1)
  conv1 = convolution_layer(32)(conv1)
  input_layers.append(conv1)
  crop1 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,3)))(conv1)
  pool1 = max_pooling_layer()(conv1)
  input_layers.append(pool1)

  conv2 = convolution_layer(64)(pool1)
  input_layers.append(conv2)
  conv2 = convolution_layer(64)(conv2)
  input_layers.append(conv2)
  crop2 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))(conv2)
  pool2 = max_pooling_layer()(conv2)
  input_layers.append(pool2)

  deconv2 = concatenated_de_convolution_layer(64)(pool2)
  input_layers.append(deconv2)
  merge2 = tf.keras.layers.concatenate([deconv2, crop2], axis=3)
  input_layers.append(merge2)
  deconv_merge2 = convolution_layer(64)(merge2)
  input_layers.append(deconv_merge2)
  deconv_merge2 = convolution_layer(64)(deconv_merge2)
  input_layers.append(deconv_merge2)

  deconv1=concatenated_de_convolution_layer(32)(deconv_merge2)
  input_layers.append(deconv1)
  merge1 = tf.keras.layers.concatenate([deconv1, crop1], axis=3)
  input_layers.append(merge1)
  deconv_merge1=convolution_layer(32)(merge1)
  input_layers.append(deconv_merge1)
  deconv_merge1=convolution_layer(32)(deconv_merge1)
  input_layers.append(deconv_merge1)

  deconv0=concatenated_de_convolution_layer(16)(deconv_merge1)
  input_layers.append(deconv0)
  merge0 = tf.keras.layers.concatenate([deconv0, crop0], axis=3)
  input_layers.append(merge0)
  deconv_merge0=convolution_layer(32)(merge0)
  input_layers.append(deconv_merge0)
  deconv_merge0=convolution_layer(32)(deconv_merge0)
  input_layers.append(deconv_merge0)

  resized=tf.keras.layers.Lambda(lambda image:
                         tf.image.resize_images(
                           images=image,
                           size=[480, 854]
                         ))(deconv_merge0)

  input_layers.append(resized)
  final_layer =  convolution_layer(1, kernel=(1, 1), activation='sigmoid')(resized)
  input_layers.append(final_layer)
  for index in range(len(input_layers)):
    logging.info("Layer {} shape:{}".format(index, input_layers[index].shape))
  return final_layer


def optimizer_init_fn(FLAGS):
  # optimizer = tf.train.MomentumOptimizer(FLAGS.lr, 0.9, use_nesterov=True)
  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  return optimizer


def unet_wo_connect(FLAGS, channel_dim, inputs):
  logging.info("In unet_wo_connect")
  input_shape = (FLAGS.height, FLAGS.weight, channel_dim)

  layer_list = []
  layer_list.append(tf.keras.layers.InputLayer(input_shape=input_shape))

  if FLAGS.layer8:
    layer_list.append(convolution_layer(8))
    layer_list.append(convolution_layer(8))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer16:
    layer_list.append(convolution_layer(16))
    layer_list.append(convolution_layer(16))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer32:
    layer_list.append(convolution_layer(32))
    layer_list.append(convolution_layer(32))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer64:
    layer_list.append(convolution_layer(64))
    layer_list.append(convolution_layer(64))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer128:
    layer_list.append(convolution_layer(128))
    layer_list.append(convolution_layer(128))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer256:
    layer_list.append(convolution_layer(256))
    layer_list.append(convolution_layer(256))
    layer_list.append(max_pooling_layer())

  if FLAGS.layer256:
    layer_list.append(concatenated_de_convolution_layer(256))
    layer_list.append(convolution_layer(256))
    layer_list.append(convolution_layer(256))

  if FLAGS.layer128:
    layer_list.append(concatenated_de_convolution_layer(128))
    layer_list.append(convolution_layer(128))
    layer_list.append(convolution_layer(128))

  if FLAGS.layer64:
    layer_list.append(concatenated_de_convolution_layer(64))
    layer_list.append(convolution_layer(64))
    layer_list.append(convolution_layer(64))

  if FLAGS.layer32:
    layer_list.append(concatenated_de_convolution_layer(32))
    layer_list.append(convolution_layer(32))
    layer_list.append(convolution_layer(32))

  if FLAGS.layer16:
    layer_list.append(concatenated_de_convolution_layer(16))
    layer_list.append(convolution_layer(16))
    layer_list.append(convolution_layer(16))

  if FLAGS.layer8:
    layer_list.append(concatenated_de_convolution_layer(8))
    layer_list.append(convolution_layer(8))
    layer_list.append(convolution_layer(8))

  resize_layer = tf.keras.layers.Lambda(lambda image:
                                        tf.image.resize_images(
                                          images=image,
                                          size=[480, 854]
                                        ))

  layer_list.append(resize_layer)
  layer_list.append(convolution_layer(1, kernel=(1, 1), activation='sigmoid'))


  logging.info("Input Layer shape: {}".format(inputs.get_shape()))
  output = []
  output.append(inputs)
  for index in range(len(layer_list)):
    output.append(layer_list[index](output[index]))
    logging.info("Layer {} shape:{}".format(index, output[index + 1].get_shape()))

  model = tf.keras.models.Sequential(layers=layer_list)
  return model(inputs=inputs)

# Not used yet
def dice_coefficient_loss(labels=None, logits=None):
  y1 = tf.contrib.layers.flatten(labels)
  y2 = tf.contrib.layers.flatten(logits)
  return 1 - ((2. * tf.reduce_sum(y1 * y2) + smoothness) / (tf.reduce_sum(y1) + tf.reduce_sum(y2) + smoothness))


def unet_w_connect128(FLAGS, channel_dim, inputs):
  input_shape = (FLAGS.height, FLAGS.weight, channel_dim)
  logging.info("In unet_w_connect128")
  input_layers=[]
  conv0 = convolution_layer(16)(inputs)
  input_layers.append(conv0)
  conv0 = convolution_layer(16)(conv0)
  input_layers.append(conv0)
  crop0 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,6)))(conv0)
  pool0 = max_pooling_layer()(conv0)
  input_layers.append(pool0)

  conv1 = convolution_layer(32)(pool0)
  input_layers.append(conv1)
  conv1 = convolution_layer(32)(conv1)
  input_layers.append(conv1)
  crop1 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,3)))(conv1)
  pool1 = max_pooling_layer()(conv1)
  input_layers.append(pool1)

  conv2 = convolution_layer(64)(pool1)
  input_layers.append(conv2)
  conv2 = convolution_layer(64)(conv2)
  input_layers.append(conv2)
  crop2 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))(conv2)
  pool2 = max_pooling_layer()(conv2)
  input_layers.append(pool2)

  conv3 = convolution_layer(128)(pool2)
  input_layers.append(conv3)
  conv3 = convolution_layer(128)(conv3)
  input_layers.append(conv3)
  crop3 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv3)
  pool3 = max_pooling_layer()(conv3)
  input_layers.append(pool3)

  deconv3 = concatenated_de_convolution_layer(128)(pool3)
  input_layers.append(deconv3)
  merge3 = tf.keras.layers.concatenate([deconv3, crop3], axis=3)
  input_layers.append(merge3)
  deconv_merge3 = convolution_layer(128)(merge3)
  input_layers.append(deconv_merge3)
  deconv_merge3 = convolution_layer(128)(deconv_merge3)
  input_layers.append(deconv_merge3)
  logging.info(deconv_merge3.shape)

  deconv2 = concatenated_de_convolution_layer(64)(deconv_merge3)
  input_layers.append(deconv2)
  merge2 = tf.keras.layers.concatenate([deconv2, crop2], axis=3)
  input_layers.append(merge2)
  deconv_merge2 = convolution_layer(64)(merge2)
  input_layers.append(deconv_merge2)
  deconv_merge2 = convolution_layer(64)(deconv_merge2)
  input_layers.append(deconv_merge2)
  logging.info(deconv_merge2.shape)

  deconv1=concatenated_de_convolution_layer(32)(deconv_merge2)
  input_layers.append(deconv1)
  merge1 = tf.keras.layers.concatenate([deconv1, crop1], axis=3)
  input_layers.append(merge1)
  deconv_merge1=convolution_layer(32)(merge1)
  input_layers.append(deconv_merge1)
  deconv_merge1=convolution_layer(32)(deconv_merge1)
  input_layers.append(deconv_merge1)

  deconv0=concatenated_de_convolution_layer(16)(deconv_merge1)
  input_layers.append(deconv0)
  merge0 = tf.keras.layers.concatenate([deconv0, crop0], axis=3)
  input_layers.append(merge0)
  deconv_merge0=convolution_layer(32)(merge0)
  input_layers.append(deconv_merge0)
  deconv_merge0=convolution_layer(32)(deconv_merge0)
  input_layers.append(deconv_merge0)

  resized=tf.keras.layers.Lambda(lambda image:
                         tf.image.resize_images(
                           images=image,
                           size=[480, 854]
                         ))(deconv_merge0)

  input_layers.append(resized)
  final_layer =  convolution_layer(1, kernel=(1, 1), activation='sigmoid')(resized)
  input_layers.append(final_layer)
  for index in range(len(input_layers)):
    logging.info("Layer {} shape:{}".format(index, input_layers[index].shape))
  return final_layer


def unet_w_connect_unet(FLAGS, channel_dim, inputs):
  logging.info("In unet_w_connect_unet")
  input_shape = (FLAGS.height, FLAGS.weight, channel_dim)
  input_layers=[]
  conv0 = convolution_layer(64)(inputs)
  input_layers.append(conv0)
  conv0 = convolution_layer(64)(conv0)
  input_layers.append(conv0)
  crop0 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,6)))(conv0)
  pool0 = max_pooling_layer()(conv0)
  input_layers.append(pool0)

  conv1 = convolution_layer(128)(pool0)
  input_layers.append(conv1)
  conv1 = convolution_layer(128)(conv1)
  input_layers.append(conv1)
  crop1 = tf.keras.layers.Cropping2D(cropping=((0,0),(0,3)))(conv1)
  pool1 = max_pooling_layer()(conv1)
  input_layers.append(pool1)

  conv2 = convolution_layer(256)(pool1)
  input_layers.append(conv2)
  conv2 = convolution_layer(256)(conv2)
  input_layers.append(conv2)
  crop2 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))(conv2)
  pool2 = max_pooling_layer()(conv2)
  input_layers.append(pool2)

  conv3 = convolution_layer(512)(pool2)
  input_layers.append(conv3)
  conv3 = convolution_layer(512)(conv3)
  input_layers.append(conv3)
  crop3 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv3)
  pool3 = max_pooling_layer()(conv3)
  input_layers.append(pool3)

  conv4 = convolution_layer(1024)(pool3)
  conv4 = convolution_layer(1024)(conv4)

  deconv3 = concatenated_de_convolution_layer(512)(conv4)
  input_layers.append(deconv3)
  merge3 = tf.keras.layers.concatenate([deconv3, crop3], axis=3)
  input_layers.append(merge3)
  deconv_merge3 = convolution_layer(512)(merge3)
  input_layers.append(deconv_merge3)
  deconv_merge3 = convolution_layer(512)(deconv_merge3)
  input_layers.append(deconv_merge3)
  logging.info(deconv_merge3.shape)

  deconv2 = concatenated_de_convolution_layer(256)(deconv_merge3)
  input_layers.append(deconv2)
  merge2 = tf.keras.layers.concatenate([deconv2, crop2], axis=3)
  input_layers.append(merge2)
  deconv_merge2 = convolution_layer(256)(merge2)
  input_layers.append(deconv_merge2)
  deconv_merge2 = convolution_layer(256)(deconv_merge2)
  input_layers.append(deconv_merge2)
  logging.info(deconv_merge2.shape)

  deconv1=concatenated_de_convolution_layer(128)(deconv_merge2)
  input_layers.append(deconv1)
  merge1 = tf.keras.layers.concatenate([deconv1, crop1], axis=3)
  input_layers.append(merge1)
  deconv_merge1=convolution_layer(128)(merge1)
  input_layers.append(deconv_merge1)
  deconv_merge1=convolution_layer(128)(deconv_merge1)
  input_layers.append(deconv_merge1)

  deconv0=concatenated_de_convolution_layer(64)(deconv_merge1)
  input_layers.append(deconv0)
  merge0 = tf.keras.layers.concatenate([deconv0, crop0], axis=3)
  input_layers.append(merge0)
  deconv_merge0=convolution_layer(64)(merge0)
  input_layers.append(deconv_merge0)
  deconv_merge0=convolution_layer(64)(deconv_merge0)
  input_layers.append(deconv_merge0)

  resized=tf.keras.layers.Lambda(lambda image:
                         tf.image.resize_images(
                           images=image,
                           size=[480, 854]
                         ))(deconv_merge0)

  input_layers.append(resized)
  final_layer =  convolution_layer(1, kernel=(1, 1), activation='sigmoid')(resized)
  input_layers.append(final_layer)
  for index in range(len(input_layers)):
    logging.info("Layer {} shape:{}".format(index, input_layers[index].shape))
  return final_layer