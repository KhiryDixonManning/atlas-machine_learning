#!/usr/bin/env python3
""" module containing class that represents the YOLOv3 algorithm """
import numpy as np
import tensorflow as tf


class Yolo():
    """ class that represents the YOLOv3 algorithm to perform object detection

            METHODS
            =======


            PUBLIC INSTANCE ATTRIBUTES
            ==========================
            model [model]: the Darknet Keras model

            class_names [list]: a list of the class names for the model

            class_t [float]: the box score threshold for the initial
            filtering step

            nms_t [float]: the IOU threshold for non-max suppression

            anchors [np.ndarry]: the anchor boxes

    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ initializer for YOLO class

                PARAMETERS
                ==========
                    model_path [str]: path to where a Darknet Keras model is
                    stored

                    classes_path [str]: path to where the list of class names
                    used for the Darknet model, listed in
                    order of index, can be found

                    class_t [float]: the box score threshold for the initial
                    filtering step

                    nms_t [float]: the IOU threshold for non-max suppression

                    anchors [np.ndarry]: all of the anchor boxes,
                    shape (outputs, anchor_boxes, 2)
                        - outputs [int]: # of outputs (predictions) made by
                            the Darknet model
                        - anchor_boxes [int]: # of anchor boxes used for each
                            prediction
                        - 2 => [anchor_box_width, anchor_box_height


                PUBLIC INSTANCE ATTRIBUTES
                ==========================
                model [model]: the Darknet Keras model

                class_names [list]: a list of the class names for the model

                class_t [float]: the box score threshold for the initial
                filtering step

                nms_t [float]: the IOU threshold for non-max suppression

                anchors [np.ndarry]: the anchor boxes


        """
        self.model = tf.keras.models.load_model(model_path)
        class_txt = open(classes_path, 'r')
        classes = class_txt.read()
        classes_list = classes.replace('\n', '.').split('.')
        if '' in classes_list:
            classes_list.remove('')
        self.class_names = classes_list
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
