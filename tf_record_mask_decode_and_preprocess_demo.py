# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import tensorflow as tf

import sys
import os
import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt
# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/robin/eclipse-workspace-python/TF_models/models/research"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy
from deeplab.utils import get_dataset_colormap


slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags


# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.



# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 2,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.


flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', "/home/robin/Dataset/VOC/VOC2012_VOCtrainval/sematic_segmentation_tfrecord", 'Where the dataset reside.')
FLAGS = flags.FLAGS


def write_file(file_name_string,seg):
    with open(file_name_string, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        for i in range(seg.shape[0]):
            spamwriter.writerow(seg[i][:])

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)

 

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      samples = input_generator.get(
          dataset,
          FLAGS.train_crop_size,
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=128 * config.num_clones)
      
      
      samples = inputs_queue.dequeue()

      # Add name to input and label nodes so we can add to summary.
      samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
      samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)
      
      print(samples)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()
      
    
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        print('Start verification process...')
        try:
          while True:
            out_image, out_label  = session.run([samples[common.IMAGE],samples[common.LABEL]])
            
            #write_file("out_label.csv",np.squeeze(out_label[0], axis=2))
            
            cv2.imshow('out_image',cv2.cvtColor(out_image[0]/255,cv2.COLOR_RGB2BGR))
            cv2.imshow('out_label',np.asarray(out_label[0]*100, dtype=np.uint8))

            
            
            colored_label = get_dataset_colormap.label_to_color_image(np.squeeze(out_label[0]), dataset=get_dataset_colormap.get_pascal_name())
            cv2.imshow("colored_label",cv2.cvtColor(colored_label.astype(np.uint8),cv2.COLOR_RGB2BGR))
           
            alpha = 0.5
            img_add = cv2.addWeighted(out_image[0], alpha, colored_label.astype(np.float32), 1-alpha, 0)
            cv2.imshow("colored_overlap",cv2.cvtColor(img_add,cv2.COLOR_RGB2BGR)/255)
            cv2.waitKey(0)
            

        except tf.errors.OutOfRangeError:
          print("end!") 

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':

  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
