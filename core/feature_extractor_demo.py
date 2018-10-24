

"""Tests for xception.py."""
import numpy as np
import six
import tensorflow as tf
import time
from datetime import datetime

import sys
import os
# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/robin/eclipse-workspace-python/TF_models/models/research"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from deeplab.core import feature_extractor

slim = tf.contrib.slim




flags = tf.app.flags

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')

# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
# See core/feature_extractor.py for supported model variants.
flags.DEFINE_string('model_variant', 'xception_65', 'DeepLab model variant.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')




FLAGS = flags.FLAGS


if __name__ == '__main__':
    
    images = tf.random_normal([1, 513, 513, 3])


    features, end_points = feature_extractor.extract_features(
          images,
          output_stride=FLAGS.output_stride,
          multi_grid=FLAGS.multi_grid,
          model_variant=FLAGS.model_variant,
          depth_multiplier=FLAGS.depth_multiplier,
          weight_decay=0.0001,
          reuse=None,
          is_training=False,
          fine_tune_batch_norm=False)
    
    
    print(features, end_points)
    
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    
    
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape())) 
    
    
    
    
    
    