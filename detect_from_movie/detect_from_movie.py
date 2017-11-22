import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import progressbar

# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_08.tar.gz
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = '../model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
# RFCN_GRAPH_FILE = '../model/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
# FASTER_RCNN_GRAPH_FILE = '../model/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

LABEL_FILE = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
TEST_IMAGES_PATH = './test_images'

INPUT_FILE = 'movie.m4v'
OUTPUT_FILE = 'output.m4v'


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

print('prepare sesssion')

# --- prepare object detection
detection_graph = load_graph(SSD_GRAPH_FILE)
# detection_graph = load_graph(RFCN_GRAPH_FILE)
# detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# --- load labels
label_map = label_map_util.load_labelmap(LABEL_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# --- about input video
video_in = cv2.VideoCapture(INPUT_FILE)
frame_count = int(video_in.get(7))
frame_rate = int(video_in.get(5))

# --- output video
fps = 1000 / frame_rate
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (960, 540))

print('start process frames...')

# --- progressbar
bar = progressbar.ProgressBar(max_value=frame_count)

for i in range(frame_count):
    bar.update(i)

    with tf.Session(graph=detection_graph) as sess:

        # frame.shape : [height, width, 3]
        _, frame = video_in.read()

        # frame_np_expanded.shape : [1, height, width, 3]
        frame_np_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                            feed_dict={image_tensor: frame_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=5)

        # --- save
        video_out.write(frame)

        # # --- show images
        # plt.figure(figsize=(12, 8))
        # plt.imshow(frame)
        # plt.show()

video_in.release()
video_out.release()
