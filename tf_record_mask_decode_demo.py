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

import numpy as np
import PIL.Image as img

from matplotlib import gridspec
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
from deeplab.utils import get_dataset_colormap
from deployment import model_deploy
from deeplab.core import preprocess_utils
from deeplab import input_preprocess

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

# Dataset settings.
flags.DEFINE_string('dataset', 'ade20k',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', "/home/robin/Dataset/ADE20K/semantic_segmentation_tfrecord", 'Where the dataset reside.')
FLAGS = flags.FLAGS

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(16, 8))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  plt.imshow(seg_map)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_map, alpha=0.8)
  plt.axis('off')
  plt.title('segmentation overlay')
  plt.show()

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
  
  data_provider  = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=3,
      common_queue_capacity=20 * 1,
      common_queue_min=10 * 1,
      shuffle=False)
  image, label, image_name, height, width = input_generator.get_decode_data(data_provider,
                                                      FLAGS.train_split)
  print(image, label, image_name, height, width)

  original_image, processed_image, label = input_preprocess.preprocess_image_and_label(
      image,
      label,
      crop_height=513,
      crop_width=513,
      min_resize_value=513,
      max_resize_value=513,
      resize_factor=None,
      min_scale_factor=0.5,
      max_scale_factor=2,
      scale_factor_step_size=0.25,
      ignore_label=0,
      is_training=True,
      model_variant="mobilenet_v2")

  
  init=tf.global_variables_initializer()
  with tf.Session() as session:
        session.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        print('Start verification process...')
        for l in range(data_provider._num_samples):
            out_image, out_label, out_image_name, out_height, out_width = session.run([processed_image, label, image_name, height, width])

#             print(out_label.shape)
#             print(out_image ,out_label.shape ,out_height, out_width)
#             print(out_image, out_label, out_image_name, out_height, out_width)
             
            
#             vis_segmentation(out_image/255, np.squeeze(out_label, axis=2))
            
            colored_label = get_dataset_colormap.label_to_color_image(np.squeeze(out_label, axis=2), dataset=get_dataset_colormap.get_ade20k_name())
            colored_label_uint8 = np.asarray(colored_label, dtype=np.uint8)
            cv2.imshow("colored_label",cv2.cvtColor(colored_label_uint8,cv2.COLOR_RGB2BGR))
     
     
            colored_label = colored_label.astype(np.float32) #np.asarray(colored_label, dtype=np.float32)
            alpha = 0.3
            img_add = img_add = cv2.addWeighted(out_image, alpha, colored_label, 1-alpha, 0)
            cv2.imshow("colored_overlap",cv2.cvtColor(img_add,cv2.COLOR_RGB2BGR)/255)
            cv2.waitKey(0)
            
            
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
#   flags.mark_flag_as_required('train_logdir')
#   flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
