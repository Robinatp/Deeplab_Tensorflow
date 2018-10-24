import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import time
from datetime import datetime

slim = tf.contrib.slim


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

  def __init__(self, modir_dir):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    with tf.gfile.GFile(os.path.join(modir_dir,self.FROZEN_GRAPH_NAME), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())  

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
      
    self.sess = tf.Session(graph=self.graph)
    
    ops = self.sess.graph.get_operations()
    for op in ops:
            print(op.name)
            

    writer = tf.summary.FileWriter("./logs", graph=self.graph)
    writer.close()

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    print("origin_size:",image.size,"  target_size:",target_size)
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    
    start_time = time.time()
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    duration = time.time() - start_time
          
    print ('%s: , duration = %.3f s ' %(datetime.now(), duration))
    
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()

def run_visualization(deeplab,image_dir):
  """Inferences DeepLab model and visualizes result."""
  
  image_files = tf.gfile.Glob(image_dir+"*.jpg")
  print(image_files)
  
  for file in image_files:
      with tf.gfile.FastGFile(file) as f:
          original_im = Image.open(BytesIO(f.read()))
      
      resized_im, seg_map = MODEL.run(original_im)
    
#       vis_segmentation(resized_im, seg_map)
      
      image_raw =  cv2.imread(file)
      image_resize = cv2.resize(image_raw,resized_im.size)
      cv2.imshow('image_raw',image_resize)
      
      colored_label = label_to_color_image(seg_map)
      colored_label = cv2.cvtColor(colored_label.astype(np.uint8),cv2.COLOR_RGB2BGR)
      cv2.imshow("colored_label",colored_label)
            
      alpha = 0.4
      img_add = img_add = cv2.addWeighted(image_resize, alpha, colored_label, 1-alpha, 0)
      cv2.imshow("colored_overlap",img_add)
      cv2.waitKey(0)



MODEL_DIR= "/home/robin/Dataset/models/semantic_segmentation/deeplabv3_pascal_trainval_2018_01_04/deeplabv3_pascal_trainval"
# MODEL_DIR= "/home/robin/Dataset/models/semantic_segmentation/deeplabv3_mnv2_pascal_train_aug_2018_01_29/deeplabv3_mnv2_pascal_train_aug"

flags = tf.app.flags

# Dataset settings.
flags.DEFINE_string('dataset', 'ade20k',
                    'Name of the segmentation dataset.')

tf.app.flags.DEFINE_string(
    'test_path', 'images_demo/', 'Test image path.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('modir_dir', MODEL_DIR, 'Where the Model reside.')

FLAGS = flags.FLAGS

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


if __name__ == "__main__":
    MODEL = DeepLabModel(FLAGS.modir_dir)
    print('model loaded successfully!')
    
    run_visualization(MODEL,FLAGS.test_path)
    
    
    
    