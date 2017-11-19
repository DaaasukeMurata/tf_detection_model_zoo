import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import glob
import os.path
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_08.tar.gz
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

# -- In[22]
# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = '../model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
# SSD_GRAPH_FILE = '../model/ssd_mobilenet_v1_coco_2017_11_08/frozen_inference_graph.pb'
# RFCN_GRAPH_FILE = '../model/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
# FASTER_RCNN_GRAPH_FILE = '../model/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

LABEL_FILE = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
TEST_IMAGES_PATH = './test_images'


def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# -- In[24]
detection_graph = load_graph(SSD_GRAPH_FILE)
# detection_graph = load_graph(RFCN_GRAPH_FILE)
# detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

### load labels
label_map = label_map_util.load_labelmap(LABEL_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# fnamesは、ファイル名のリストとなる
fnames = glob.glob(TEST_IMAGES_PATH + '/*.jpg')
for fname in fnames:
    with tf.Session(graph=detection_graph) as sess:
        print("processing...  " + fname)
        image = Image.open(fname)
        # image_np.shape : [height, width, 3]
        image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded.shape : [1, height, width, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                            feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=5)

        ### save jpg file
        image_pil = Image.fromarray(image_np)
        # os.path.splitext(./test_img/sample1.jpg) -> './test_img/sample1', '.jpg'
        body_name, ext_name = os.path.splitext(fname)
        outname = body_name + "_detected" + ext_name
        image_pil.save(outname)

        # plt.figure(figsize=(12, 8))
        # plt.imshow(image_np)
        # plt.show()
