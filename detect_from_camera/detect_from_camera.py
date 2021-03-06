import tensorflow as tf
import numpy as np
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = '../model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
# FASTER_RCNN_GRAPH_FILE = '../model/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb'


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


if __name__ == '__main__':

    print('prepare sesssion')

    # --- prepare object detection
    detection_graph = load_graph(SSD_GRAPH_FILE)
    # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # --- load labels
    label_map = label_map_util.load_labelmap(LABEL_FILE)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # --- input video
    # 0 or 1 is device number. On mac, 0 is internal camera, 1 is USB camera.
    cap = cv2.VideoCapture(1)

    print('start process camera frames...')

    while(True):

        with tf.Session(graph=detection_graph) as sess:

            # frame.shape : [height, width, 3]
            cap_ret, frame = cap.read()
            if cap_ret != True:
                continue

            # # # resize for 処理負荷軽減
            # RESIZE_RATE = 1 / 4
            # frame = cv2.resize(frame, None, fx=RESIZE_RATE, fy=RESIZE_RATE)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # frame_np_expanded.shape : [1, height, width, 3]
            frame_np_expanded = np.expand_dims(frame, axis=0)

            # time
            start_time = time.time()

            (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                                feed_dict={image_tensor: frame_np_expanded})

            # time
            print('elapse_time1 : {0}'.format(time.time() - start_time))

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)

            # --- show output
            cv2.imshow('detect', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
