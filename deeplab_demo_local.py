#@title Imports

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import csv
import cv2
import tensorflow as tf


#@title Helper methods


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = ['SemanticPredictions:0',"strided_slice:0"]
  INPUT_SIZE = 513

  def __init__(self, frozen_graph):
    """Creates and loads pretrained deeplab model."""
    self.graph = self.load_graph(frozen_graph)

    self.sess = tf.Session(graph=self.graph)
    
  def load_graph(self, frozen_graph):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        seg_tensor = tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=['SemanticPredictions:0'], 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )   
        
        writer = tf.summary.FileWriter("./logs_graph", graph=graph)
        writer.close() 
        
    return graph

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    print("ori:",image.size)
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    print("target:",resized_image)
    batch_seg_map,process_image = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    print("seg:",seg_map.shape)
    print("process:",process_image)
    
    upsampled_output = tf.image.resize_nearest_neighbor(
        tf.expand_dims(batch_seg_map, axis=-1), [height, width])
    upsampled_seg = tf.squeeze(upsampled_output)
    upsampled_seg = tf.cast(upsampled_seg,tf.uint8)
    with tf.Session() as sess:
        upsampled_seg = sess.run(upsampled_seg)
    print("upsampled_seg",upsampled_seg.shape)
    
    
    return resized_image, upsampled_seg#np.squeeze(process_image+1, axis=0), seg_map


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


def vis_segmentation(ori_image, image, seg_map):
  """
  Visualizes input image, segmentation map and overlay view.
  ori_image, original image
  image, resize image adapt to 513
  seg_map, finale segment with size 513*513
  
  """
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  print(ori_image)
  plt.imshow(ori_image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(ori_image)
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
  





def run_visualization(DeepLabModel,path):
  """Inferences DeepLab model and visualizes result."""
  orignal_im = Image.open(path)
  resized_im, seg_map = DeepLabModel.run(orignal_im)
  write_file("mask.csv",seg_map)
  
  
#   image =cv2.imread(path)
#   image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#   
#   image=cv2.add(image_RGB, np.zeros(np.shape(image_RGB), dtype=np.uint8), mask=seg_map)
#   cv2.imwrite("demo_extractor.jpg", image)
#   cv2.imshow('mask', image)
#   cv2.waitKey(0)
        
        
  vis_segmentation(orignal_im, resized_im, seg_map)


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
    
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def write_file(file_name_string,seg):
    with open(file_name_string, 'wb') as csvfile:
     
        spamwriter = csv.writer(csvfile, dialect='excel')
        for i in range(seg.shape[0]):
            spamwriter.writerow(seg[i][:])


def main():

    BASE_PATH = 'image_battery'
    TEST_IMAGES = os.listdir(BASE_PATH)
    TEST_IMAGES.sort()
    print(TEST_IMAGES)
    
    #MODEL = DeepLabModel("datasets/battery_word_seg/exp_mobilenetv2/train_on_trainval_set/export/frozen_inference_graph.pb")
    MODEL = DeepLabModel("datasets/battery_seg/exp_mobilenetv2/train_on_trainval_set/export/frozen_inference_graph.pb")
    #MODEL = DeepLabModel("datasets/pascal_voc_seg/exp_mobilenetv2/train_on_trainval_set/export/frozen_inference_graph.pb")
    #MODEL = DeepLabModel("datasets/cityscapes/init_models/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb")
    #MODEL = DeepLabModel("datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/frozen_inference_graph.pb")
    #MODEL = DeepLabModel("datasets/pascal_voc_seg/exp_xcption/train_on_trainval_set/export/frozen_inference_graph.pb")
    
    print('model loaded successfully!')
    
    for image in TEST_IMAGES:
        image_path = os.path.join(BASE_PATH, image)
        run_visualization(MODEL,image_path)
    
main()



