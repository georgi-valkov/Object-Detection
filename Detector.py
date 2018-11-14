import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2



class Detector():
    def __init__(self, graph, labels, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.graph = self.__load_graph(graph)
        self.category_index = label_map_util.create_category_index_from_labelmap(labels)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(graph=self.graph, config=self.config)




    def __load_graph(self, path_to_frozen_graph):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect(self, frame, resizing_factor):
        # Keeping the original frame
        original_image = frame
        # Resize image for faster detection
        height = frame.shape[0]
        frame = image_resize(frame, height=int(height/resizing_factor))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection.
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.session.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})
        # Applying visual utilities to the image array (green boxes, labels, confidence score)
        vis_util.visualize_boxes_and_labels_on_image_array(
            original_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=5)

        return original_image, scores, num_detections, boxes

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)


    # return the resized image
    return resized