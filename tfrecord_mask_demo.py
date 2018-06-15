#coding=utf-8
import tensorflow as tf
import sys
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv

# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/ubuntu/eclipse-workspace/models/research/"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
slim = tf.contrib.slim


NUM_CLASSES = 20
SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

labels_to_class =['none','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FILE_PATTERN = 'voc_2007_%s_*.tfrecord'

def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/%s*.tfrecord' % (dataset_dir, split_name)

def bboxes_draw_on_img(img, classes, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], colors, thickness)
        # Draw text...
        s = '%s' % (labels_to_class[classes[i]])
        p1 = (p1[0]+15, p1[1]+5)
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, colors, 1)
        
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.
    Returns:
      A `Dataset` namedtuple.
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    
    if not file_pattern:
        file_pattern = FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
#     #文件名格式
#     if file_pattern is None:
#         file_pattern = _get_output_filename('tfrecords','voc_2007_train')#need fix your filename
#     print(file_pattern)
    
    # 适配器1：将example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    
    #适配器2：将反序列化的数据组装成更高级的格式。由slim完成
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    # 解码器
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    
    # dataset对象定义了数据集的文件位置，解码方式等元信息
    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=reader,
                num_samples = SPLITS_TO_SIZES['test'], # 手动生成了三个文件， 每个文件里只包含一个example
                decoder=decoder,
                items_to_descriptions = ITEMS_TO_DESCRIPTIONS,
                num_classes=NUM_CLASSES)
    return dataset

#读取tfrecords文件
def decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    example=tf.parse_single_example(serialized,features={
        'image/height':tf.FixedLenFeature([],tf.int64),
        'image/width':tf.FixedLenFeature([],tf.int64),
        'image/encoded':tf.FixedLenFeature([],tf.string),
        'image/object/class/label':tf.FixedLenFeature([],tf.int64)
    })
    label=tf.cast(example['image/object/class/label'], tf.int32)
    image=tf.decode_raw(example['image/encoded'],tf.uint8)
    image=tf.reshape(image,tf.stack([
        tf.cast(example['image/height'], tf.int32),
        tf.cast(example['image/width'], tf.int32),
        3]))
    
    print('decode_from_tfrecords: ',image)  
    print('decode_from_tfrecords: ',label)
    return image,label

def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
#             crop_img = img[xmin:(xmax - xmin),xmax:(ymax - ymin)]
#             misc.imsave('1.jpg', crop_img)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = CLASSES[cls_id]
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()

def write_file(file_name_string,seg):
    with open(file_name_string, 'wb') as csvfile:
     
        spamwriter = csv.writer(csvfile, dialect='excel')
        for i in range(seg.shape[0]):
            spamwriter.writerow(seg[i][:])
def test():
    reconstructed_images = []
    record_iterator = tf.python_io.tf_record_iterator(path='datasets/battery_word_seg/tfrecord/trainval-00000-of-00002.tfrecord')
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()     
        threads= tf.train.start_queue_runners(coord=coord)
       
        
        for string_iterator in record_iterator:
            plt.figure(figsize=(12, 12))
            example = tf.train.Example()
            example.ParseFromString(string_iterator)
            height = example.features.feature['image/height'].int64_list.value[0]
            width = example.features.feature['image/width'].int64_list.value[0]
            png_string = example.features.feature['image/encoded'].bytes_list.value[0]
#             label = example.features.feature['image/object/class/label'].int64_list.value[0]
#             xmin = example.features.feature['image/object/bbox/xmin'].float_list.value[0]
#             xmax = example.features.feature['image/object/bbox/xmax'].float_list.value[0]
#             ymin = example.features.feature['image/object/bbox/ymin'].float_list.value[0]
#             ymax = example.features.feature['image/object/bbox/ymax'].float_list.value[0]
            
            encoded_mask_string = example.features.feature['image/segmentation/class/encoded'].bytes_list.value[0]
            
            plt.subplot(131)
            mask_decode_png = tf.image.decode_png(encoded_mask_string, channels=1)
            fix_mask =tf.cast(tf.greater(mask_decode_png,0),tf.uint8)
           
            
            redecode_mask_img = sess.run(mask_decode_png)
#             write_file("mask.csv",redecode_mask_img)
            print(redecode_mask_img.shape)
            redecode_mask = redecode_mask_img * 255
            mask_img = np.squeeze(redecode_mask, axis = 2)
            plt.imshow(mask_img)
            im = Image.fromarray(mask_img)
            im.save("pets.png")

            
            plt.subplot(132)
            decoded_img = tf.image.decode_jpeg(png_string, channels=3)
            reconstructed_img = sess.run(decoded_img)
            print(reconstructed_img.shape)
            plt.imshow(reconstructed_img)
           
        
        
            plt.subplot(133)
            vis_util.draw_mask_on_image_array(
                image = reconstructed_img,
                mask = np.squeeze(sess.run(fix_mask), axis = 2),
                alpha=0.8)
            plt.imshow(reconstructed_img)
  
            plt.show()
        
        
        
        coord.request_stop()     
        coord.join(threads)
test()

