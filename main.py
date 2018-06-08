import os
import sys
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
import time
import json
import yaml
import davis
import logging
import matplotlib.pyplot as plt
from model.dataset import load_data, dimension_validation, get_channel_dim
from model.dataset_gen import load_data2
from model.segmentation_model import unet_wo_connect, optimizer_init_fn, unet_w_connect, dice_coefficient_loss, unet_w_connect128, unet_w_connect_unet
from utils.util import load_image_files, check_image_dimension, load_seq_from_yaml, path_config, write_summary, print_image, get_channel_dimension2
import tensorflow.contrib.eager as tfe
import pdb
import skimage

from davis import *

env = "cloud"
path_config(env)

tf.app.flags.DEFINE_boolean("train_mode", True, "enable training")
tf.app.flags.DEFINE_boolean("debug_mode", False, "pdb debugger")
tf.app.flags.DEFINE_boolean("skip_test_mode", False, "skip test")
tf.app.flags.DEFINE_boolean("skip_train_val_mode", True, "skip test") # default not eval on train-val


tf.app.flags.DEFINE_boolean("enable_connect_unet", True, "enable_connect_unet")
tf.app.flags.DEFINE_boolean("enable_connect128", True, "enable_connect128")
tf.app.flags.DEFINE_boolean("enable_connect", False, "enable_connect")

tf.app.flags.DEFINE_boolean("enable_osvos", True, "enable_maskrcnn")
tf.app.flags.DEFINE_boolean("enable_maskrcnn", False, "enable_maskrcnn")
tf.app.flags.DEFINE_boolean("enable_jpg", True, "enable_jpg")
tf.app.flags.DEFINE_boolean("enable_firstframe", True, "enable_firstframe")

tf.app.flags.DEFINE_string("maskrcnn_label_path", "MaskRCNN/480p", "maskrcnn_label_path")
tf.app.flags.DEFINE_string("osvos_label_path", "Results/Segmentations/480p/OSVOS2-convert", "osvos_label_path")
tf.app.flags.DEFINE_string("groundtruth_label_path", "Annotations/480p", "groundtruth_label_path")
tf.app.flags.DEFINE_string("groundtruth_image_path", "JPEGImages/480p", "groundtruth_image_path")

tf.app.flags.DEFINE_string("model_label", "", "model_label")

tf.app.flags.DEFINE_integer("height", 480, "height")
tf.app.flags.DEFINE_integer("weight", 854, "weight")
tf.app.flags.DEFINE_integer("filter", 64, "weight")
tf.app.flags.DEFINE_integer("kernel", 3, "weight")
tf.app.flags.DEFINE_integer("pad", 1, "weight")
tf.app.flags.DEFINE_integer("pool", 2, "weight")

tf.app.flags.DEFINE_integer("batch_size", 20, "batch_size")
tf.app.flags.DEFINE_integer("num_epochs", 30, "num_epochs")
tf.app.flags.DEFINE_float("lr", 0.00004, "learning rate")

tf.app.flags.DEFINE_integer("num_classes", 10, "num_classes")
tf.app.flags.DEFINE_boolean("shuffle_train", False, "shuffle")

tf.app.flags.DEFINE_boolean("layer8", False, "layer8")
tf.app.flags.DEFINE_boolean("layer16", True, "layer16")
tf.app.flags.DEFINE_boolean("layer32", True, "layer32")
tf.app.flags.DEFINE_boolean("layer64", True, "layer64")
tf.app.flags.DEFINE_boolean("layer128", False, "layer128")
tf.app.flags.DEFINE_boolean("layer256", False, "layer256")

tf.app.flags.DEFINE_integer("save_eval_every_n_epochs", 4,
                            "save prediction image for further davis score test")

tf.app.flags.DEFINE_integer("save_every_n_epochs", 4, "save model checkpoint on every n trainig epoch")

tf.app.flags.DEFINE_integer("save_train_animation_every_n_epochs", 2, "as name suggest")


FLAGS = tf.app.flags.FLAGS

root_path = os.path.join(FLAGS.output_path, FLAGS.model_label)

if not os.path.exists(root_path):
  os.makedirs(root_path)

# best model save dir
if not os.path.exists(os.path.join(root_path, "models/best/")):
  os.makedirs(os.path.join(root_path, "models/best/"))

file_handler = logging.FileHandler(root_path + "/log.txt")
file_handler.setLevel(logging.INFO)
logger = logging.getLogger('server_logger')
logger.addHandler(file_handler)


def setup(root_path):
  if not FLAGS.model_label:
    raise Exception("--model_label is required, eg osvos_maskrcnn_0602")

  if not os.path.exists(root_path + "/models"):
    os.makedirs(root_path + "/models")
    logger.info("Output path not found, create {}".format(root_path + "/models"))
  else:
    logger.info("Output Path found {}".format(root_path + "/models"))

  with open(os.path.join(root_path, "flags.json"), 'w') as fout:
    json.dump(FLAGS.flag_values_dict(), fout)
  logger.info("Flags: {}".format(FLAGS.flag_values_dict()))

  train_seqs = None
  if FLAGS.train_mode:
    train_seqs = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.train_seq_yaml))

  train_val = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.train_val_yaml))
  test_seqs = load_seq_from_yaml(os.path.join(FLAGS.config_path, FLAGS.test_val_yaml))

  return train_seqs, train_val, test_seqs


def generate_dataset(FLAGS, seqs, is_shuffle=True):
  osvos_label_paths, maskrcnn_label_paths, groundtruth_label_paths, groundtruth_image_paths, firstframe_image_paths = load_image_files(FLAGS, seqs)
  logger.info("Load {} image samples, sequences: {}".format(len(osvos_label_paths), seqs))

  if dimension_validation(
          osvos_label_paths,
          maskrcnn_label_paths,
          groundtruth_label_paths,
          groundtruth_image_paths,
          firstframe_image_paths,
          logger) is False:
    raise Exception("Invalid Image Found")

  # successfully load dataset
  segmentation_dataset, seqs = load_data2(FLAGS,
    osvos_label_paths,
    maskrcnn_label_paths,
    groundtruth_label_paths,
    groundtruth_image_paths,
    firstframe_image_paths)
  if is_shuffle:
    segmentation_dataset = segmentation_dataset.shuffle(buffer_size=1000)
  segmentation_dataset = segmentation_dataset.batch(FLAGS.batch_size)
  return segmentation_dataset, seqs


def main(unused_argv):
  train_seqs, train_sample, test_seqs = setup(root_path)
  # check_image_dimension(FLAGS, logger, train_seqs)

  # construct image files array
  if FLAGS.train_mode:
    segmentation_dataset, _ = generate_dataset(FLAGS, train_seqs, FLAGS.shuffle_train)

  segmentation_dataset_val, _ = generate_dataset(FLAGS, train_sample, False)
  segmentation_dataset_test, _ = generate_dataset(FLAGS, test_seqs, False)
  global_step = tf.Variable(0, name="global_step", trainable=False)
  exp_loss = None
  exp_dice_loss = None

  with tf.device(FLAGS.device):
    channel_dim = get_channel_dimension2(FLAGS)
    x = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, channel_dim])
    y = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.weight, 1])

    # pred_mask = model_init_fn(FLAGS=FLAGS, inputs=x)
    if FLAGS.enable_connect_unet:
      pred_mask = unet_w_connect_unet(FLAGS=FLAGS, channel_dim=channel_dim, inputs=x)
    elif FLAGS.enable_connect128:
      pred_mask = unet_w_connect128(FLAGS=FLAGS, channel_dim=channel_dim, inputs=x)
    elif FLAGS.enable_connect:
      pred_mask = unet_w_connect(FLAGS=FLAGS, channel_dim=channel_dim, inputs=x)
    else:
      pred_mask = unet_wo_connect(FLAGS=FLAGS, channel_dim=channel_dim, inputs=x)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred_mask)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_mask)

    weight = (FLAGS.height * FLAGS.weight * FLAGS.batch_size - tf.reduce_sum(y)) / tf.reduce_sum(y)
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=pred_mask, pos_weight=weight)
    tf.summary.histogram('loss_histogram', loss)

    dice_loss = dice_coefficient_loss(labels=y, logits=pred_mask)

    # osvos_dice loss
    if FLAGS.enable_osvos:
      dice_loss_osvos = dice_coefficient_loss(labels=y, logits=tf.reshape(x[:, :, :, 0], [-1, FLAGS.height, FLAGS.weight]))
    else:
      dice_loss_osvos = tf.constant(-1.0)

    loss = tf.reduce_mean(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('dice_loss', dice_loss)

    optimizer = optimizer_init_fn(FLAGS=FLAGS)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    # For tensorboard
    summary_writer = tf.summary.FileWriter(root_path+"/train", sess.graph)
    summaries = tf.summary.merge_all()

    logger.info(
      "Global number of params: {}".format(sum(v.get_shape().num_elements() for v in tf.trainable_variables())))

    best_test_dice_loss_sofar = None
    for epoch in range(FLAGS.num_epochs):
      if FLAGS.train_mode:
        logger.info("======================== Starting training Epoch {} ========================".format(epoch))
        dataset_iterator = segmentation_dataset.make_one_shot_iterator()
        batch_num = 0
        while True:
          try:
            tic = time.time()
            batch = sess.run(dataset_iterator.get_next())
            seq_name_, image_number_, object_number_, x_np, y_np = batch

            # I notice this is shape (4, 480, 854, 2) and (4, 480, 854, 1), expected?
            # further, FLAGS.batch_size==10
            logger.debug("x_np type {}, shape {}".format(type(x_np), x_np.shape))
            logger.debug("y_np type {}, shape {}".format(type(y_np), y_np.shape))
            max_label = np.max(y_np)
            if max_label >= FLAGS.num_classes:
              logger.info("WRONG! {} > num_classes".format(max_label))
              continue

            feed_dict = {x: x_np, y: y_np}

            loss_np, dice_loss_, _, pred_mask_, weight_, dice_loss_osvos_, global_step_, summaries_ = \
              sess.run([loss, dice_loss, train_op, pred_mask, weight, dice_loss_osvos, global_step, summaries], feed_dict=feed_dict)

            if not exp_loss:  # first iter
              exp_loss = loss_np
            else:
              exp_loss = 0.99 * exp_loss + 0.01 * loss_np

            if not exp_dice_loss:
              exp_dice_loss = dice_loss_
            else:
              exp_dice_loss = 0.99 * exp_dice_loss + 0.01 * dice_loss_

            if loss_np > 2.0:
              logger.info("(potentially) End of training epoch, discard last batch")
              continue

            if np.max(pred_mask_) > 1:
              print (np.where(np.max(pred_mask_, axis=(1,2,3)) > 1))

            if np.max(y_np) > 1: # very important as we are working on binary classification
              print(seq_name_)
              print(image_number_)
              print(object_number_)
              print(np.where(np.max(y_np, axis=(1,2,3)) > 1))

            assert np.max(pred_mask_) <= 1 and np.max(y_np) <= 1

            #TODO when this happens, print to see what happened
            if dice_loss_ < 0:
              print ("dice_loss should not be zero... continue")
              continue

            if epoch % FLAGS.save_train_animation_every_n_epochs == 0:
              print_image(FLAGS, seq_name_, image_number_, object_number_, x_np, y_np, pred_mask_, epoch)

            toc = time.time()

            logger.info(
              "Epoch: %i Batch: %i Train Loss: %.4f, dice loss: %.4f, smoothed loss %.4f, smoothed dice loss %.4f, "
              " dice_loss_osvos_: %4f, pos_weight: %.4f, takes %.2f seconds" %
              (epoch, batch_num, loss_np, dice_loss_, exp_loss, exp_dice_loss, dice_loss_osvos_, weight_, toc - tic)
            )
            summary_writer.add_summary(summaries_, global_step_)
            write_summary(loss_np, "Train CE Loss", summary_writer, global_step_)
            write_summary(dice_loss_, "Train Dice Loss", summary_writer, global_step_)

            write_summary(exp_loss, "Smoothed Train CE Loss", summary_writer, global_step_)
            write_summary(exp_dice_loss, "Smoothed Train Dice Loss", summary_writer, global_step_)

            # logger.info("total loss shape {}, value {}".format(total_loss_.shape, str(total_loss_)))
            batch_num += 1
          except tf.errors.OutOfRangeError:
            logger.warn("End of range")
            break

        if epoch % FLAGS.save_every_n_epochs == 0:
          if not os.path.exists(os.path.join(root_path, "models/tmp/epoch_{}/model.ckpt".format(str(epoch)))):
            os.makedirs(os.path.join(root_path, "models/tmp/epoch_{}/model.ckpt".format(str(epoch))))
          tf.train.Saver().save(sess=sess, save_path=root_path + "/models/tmp/epoch_{}/model.ckpt".format(str(epoch)))


      ##### evaluate the model on train-val and test, for EVERY EPOCH. and save best model if approriate

      for target in ["train-val", "test"]:
        if FLAGS.skip_train_val_mode and target == "train-val":
          continue

        if FLAGS.skip_test_mode and target == "test":
          continue

        seq_dataset = segmentation_dataset_val if target == "train-val" else segmentation_dataset_test
        seq_dataset = seq_dataset.make_one_shot_iterator()

        logger.info("======================== Starting testing Epoch {} - {} ========================".format(epoch, target))

        batch_num = -1
        loss_np_, dice_loss__, dice_loss_osvos__ = 0.0, 0.0, 0.0
        while True:
          try:
            batch_num += 1
            batch = sess.run(seq_dataset.get_next())
            seq_name_, image_number_, object_number_, x_np, y_np = batch

            loss_np, dice_loss_, pred_mask_, dice_loss_osvos_ = \
              sess.run([loss, dice_loss, pred_mask, dice_loss_osvos, summaries, global_step], feed_dict={x: x_np, y: y_np})

            if loss_np > 2:
              batch_num -= 1
              logger.info("(potentially) End of {} epoch, discard last batch".format(target))
              continue  # we observed that the last epoch has some wiredness due to under-rank.

            loss_np_ += loss_np
            dice_loss__ += dice_loss_
            dice_loss_osvos__ += dice_loss_osvos_

            if epoch != 0 and (epoch % FLAGS.save_eval_every_n_epochs == 0 or epoch == FLAGS.num_epochs - 1):
              # now persist prediction to disk for later davis-score computation

              test_mask_output = os.path.join(root_path, "eval", str(epoch), target)
              # /home/shared/video_segmentation/segmentation/Results/experiment_name/eval/train-val/

              for idx in range(len(seq_name_)):
                  seq_name__ = seq_name_[idx].decode("utf-8")
                  image_number__ = image_number_[idx].decode("utf-8")
                  obj_number = str(object_number_[idx])
                  savedir_ = os.path.join(test_mask_output, seq_name__, obj_number)
                  if not os.path.exists(savedir_):
                    os.makedirs(savedir_)

                  _pred_to_save = np.zeros((FLAGS.height, FLAGS.weight), dtype='uint8')
                  _pred_to_save[pred_mask_[idx, :, :, 0] > 0.5] = 1
                  davis.io.imwrite_indexed(os.path.join(savedir_, image_number__), _pred_to_save)
                
          except tf.errors.OutOfRangeError:
            logger.warn("End of range")
            break

        batch_num += 1
        loss_np_, dice_loss__, dice_loss_osvos__ = loss_np_/batch_num, dice_loss__/batch_num, dice_loss_osvos__/batch_num

        # save this batch's score to tensorboard
        summaries_, global_step_ = sess.run([summaries, global_step], feed_dict={})
        summary_writer.add_summary(summaries_, global_step_)
        write_summary(loss_np, "Test CE Loss", summary_writer, global_step_)
        write_summary(dice_loss_, "Test Dice Loss", summary_writer, global_step_)

        logger.info(
          "%s Loss: %.4f, dice loss: %.4f, dice_loss_osvos_: %4f" %
          (target, loss_np_, dice_loss__, dice_loss_osvos__)
        )

        # now check best_test_dice_loss_sofar
        if target == "test" and (best_test_dice_loss_sofar is None or dice_loss__ > best_test_dice_loss_sofar):
          # save model
          best_test_dice_loss_sofar = dice_loss__
          logger.info(
            "saving best model: Epoch %i, test_dice_loss %.4f" %
            (epoch, best_test_dice_loss_sofar)
          )
          tf.train.Saver().save(sess=sess, save_path=root_path + "/models/best/model.ckpt")



def eval_on_test_data(sess, segmentation_dataset_test, test_seq_list, ops, placeholder, epoch, FLAGS):
  # as of V2 on generator, test_seq_list is no longer in use.
  tic = time.time()
  [x, y] = placeholder
  dataset_iterator_test = segmentation_dataset_test.make_one_shot_iterator()
  test_loss, davis_j, davis_f, test_n = 0.0, {}, {}, 0
  seq_name_iter = iter(test_seq_list)

  test_mask_output = os.path.join(root_path, "unet")
  # save predicted mask to somewhere
  frame_number_by_seq_name = {}
  while True:
    try:
      batch = sess.run(dataset_iterator_test.get_next())
      seq_name, image_number, object_number, x_np, y_np = batch

       # it seems x_np has shape (2, 480, 854, 2) and y_np has shape (2, 480, 854, 1)
       # this is a little unintuitive as we expect it to be batch_size
      feed_dict = {x: x_np, y: y_np}
      test_loss_, pred_test = sess.run(ops, feed_dict=feed_dict)

      N_, H_, W_, C_ = pred_test.shape
      logging.info("pred_test.shape=={},{},{},{}".format(str(N_), str(H_), str(W_), str(C_)))
      # #TODO
      # if np.sum(pred_test> 1) > 0:
      #   logging.info("pred_test has {} >0".format(np.sum(pred_test> 1)))
      for i in range(N_):
        seq_name = next(seq_name_iter)
        seq_number = frame_number_by_seq_name.get(seq_name, 0)
        frame_number_by_seq_name[seq_name] = seq_number + 1
        # import pdb; pdb.set_trace()
        mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
        # print mask_output_dir
        if not os.path.exists(mask_output_dir):
          os.makedirs(mask_output_dir)

        mask_output = "{}/{:05d}.png".format(mask_output_dir, seq_number)

        #  pred_test is now (N_, H_, W_, num_class), we convert it to (H_, W_)
        # upon observation, turns out pred_test usually have very large 0 prediction equally large as other prediction.
        # to prevent constant 0 prediction, we reverse the list
        pred_test_ = pred_test[i, :, :, ::-1]
        base_image = np.squeeze(np.argmax(pred_test_, axis=-1))
        base_image = -base_image + FLAGS.num_classes - 1
        # above 3 line equlivalent to base_image = np.squeeze(np.argmax(pred_test[i, :, :, :], axis=-1))

        base_image = base_image.astype(np.uint8)
        io.imwrite_indexed(mask_output, base_image)
        if len(np.unique(base_image)) == 1:
          logger.info("problem on predicted base_iamge. maybe all 0 {}".format(mask_output))

      test_n += 1
      test_loss += test_loss_
    except tf.errors.OutOfRangeError:
      logger.warn("End of test range")
      break

  # not finish yet! eval davis performance
  unique_seq_name = set(test_seq_list)
  davis_j_mean, davis_f_mean = 0.0, 0.0
  for seq_name in unique_seq_name:
    mask_output_dir = "{}/{}/{}/{}".format(test_mask_output, seq_name, str(epoch), seq_name)
    sg = Segmentation(mask_output_dir, False)

    ground_truth_dir_ = "{}/{}/{}".format(FLAGS.read_path, FLAGS.groundtruth_label_path, seq_name)
    ground_truth = Segmentation(ground_truth_dir_, False)

    davis_j[seq_name] = db_eval_sequence(sg, ground_truth, measure="J", n_jobs=32)
    davis_f[seq_name] = db_eval_sequence(sg, ground_truth, measure="F", n_jobs=32)
    davis_j_mean += davis_j[seq_name]["mean"][0]
    davis_f_mean += davis_f[seq_name]["mean"][0]

  toc = time.time()
  if test_n == 0:
    test_n = 1
  return test_loss / test_n, davis_j, davis_f, \
         davis_j_mean / len(unique_seq_name), davis_f_mean / len(unique_seq_name), toc - tic


def eval_on_test_data2(sess, segmentation_dataset_test, test_seq_list, ops, placeholder, epoch, FLAGS):
  # for the per-object-based method in Approach 2, its hard to evaluate davis loss on the fly. instead, we save
  # each image on to disk and later run a seperate script to combine/post-process them
  raise NotImplementedError


if __name__ == "__main__":
  tf.app.run()
