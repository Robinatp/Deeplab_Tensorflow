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
                           "/home/robin/Dataset/VOC/VOC2012_VOCtrainval/VOC2012/JPEGImages",
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('semantic_segmentation_folder',
                           "/home/robin/Dataset/VOC/VOC2012_VOCtrainval/VOC2012/SegmentationClassRaw",
                           'Folder containing semantic segmentation annotations.')

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

def write_file(file_name_string,seg):
    with open(file_name_string, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        for i in range(seg.shape[0]):
            spamwriter.writerow(seg[i][:])

def main(unused_argv):
    
  if(FLAGS.convert):

    annotations = glob.glob(os.path.join(FLAGS.semantic_segmentation_folder,
                                       '*.' + FLAGS.segmentation_format))
    
    
            
    
    for annotation in annotations:
        print(annotation)
        
        
        ori_filename = os.path.join(FLAGS.original_color_folder,os.path.basename(annotation)[:-4]+".jpg")
        print(ori_filename)
#         ori_im =Image.open(ori_filename)
        color_im = cv2.imread(ori_filename)
        rgb_image = cv2.cvtColor(color_im,cv2.COLOR_BGR2RGB)
        print(rgb_image.shape)
         
        seg_im = cv2.imread(annotation,0)
        print(seg_im.shape)
        
        #dst = src1 * alpha + src2 * beta + gamma;
        #alpha,beta,gamma
#         alpha = 0.3
#         beta = 1-alpha
#         gamma = 0
#         img_add = cv2.addWeighted(rgb_image, alpha, seg_im, beta, gamma)
#         cv2.imshow("image_add",img_add)
#         cv2.waitKey(0)
        
         
        vis_segmentation(rgb_image,seg_im*125)


if __name__ == '__main__':
  tf.app.run()
