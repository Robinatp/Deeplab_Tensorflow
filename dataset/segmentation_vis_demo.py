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

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np

from PIL import Image

import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
import cv2
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_color_folder',
                           './battery_word_seg/JPEGImages',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('original_gt_folder',
                           './battery_word_seg/SegmentationClassRaw',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.app.flags.DEFINE_string('segmentation_output_dir',
                           './battery_word_seg/SegmentationClassRaw',
                           'folder to save modified ground truth annotations.')
tf.app.flags.DEFINE_bool('convert', True,
                           'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.

  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.gfile.Open(filename, mode='w') as f:
    pil_image.save(f, 'PNG')

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
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

def write_file(file_name_string,seg):
    with open(file_name_string, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        for i in range(seg.shape[0]):
            spamwriter.writerow(seg[i][:])

def main(unused_argv):
    
  if(FLAGS.convert):

    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                       '*.' + FLAGS.segmentation_format))
    
    
            
    
    for annotation in annotations:
        print(annotation)
        #filename = os.path.join(FLAGS.segmentation_output_dir,os.path.basename(annotation)[:-4]+".jpg")
        
        ori_filename = os.path.join(FLAGS.original_color_folder,os.path.basename(annotation)[:-4]+".jpg")
        print(ori_filename)
#         ori_im =Image.open(ori_filename)
        orignal_im = cv2.imread(ori_filename)
        image_RGB = cv2.cvtColor(orignal_im,cv2.COLOR_BGR2RGB)
         
        mask_im = cv2.imread(annotation)
        print(mask_im.shape)
        mask_RGB = cv2.cvtColor(mask_im,cv2.COLOR_BGR2RGB)
         
        vis_segmentation(orignal_im,mask_RGB)


if __name__ == '__main__':
  tf.app.run()
