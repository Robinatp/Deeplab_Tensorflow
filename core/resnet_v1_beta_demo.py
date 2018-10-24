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

from deeplab.core import resnet_v1_beta
from tensorflow.contrib.slim.nets import resnet_utils

slim = tf.contrib.slim





if __name__ == '__main__':
    inputs = tf.random_normal([1, 224, 224, 3])
    
    
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
    
        net, end_points = resnet_v1_beta.resnet_v1_101(inputs,
                 num_classes=100,
                 is_training=False,
                 global_pool=True,
                 output_stride=None,
                 multi_grid=None,
                 reuse=None,
                 scope='resnet_v1_101')
   
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    
    
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
#     print("Parameters")
#     for v in slim.get_model_variables():
#         print('name = {}, shape = {}'.format(v.name, v.get_shape())) 
    
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(10):
            start_time = time.time()
            _ = sess.run(net)
            duration = time.time() - start_time
            print ('%s: step %d, duration = %.3f' %(datetime.now(), i, duration))
            
            
