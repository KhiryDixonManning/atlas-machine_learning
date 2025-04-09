#!/usr/bin/env python3
""" module containing class that represents the YOLOv3 algorithm """
import numpy as np
import tensorflow as tf


class Yolo():
    """ class that represents the YOLOv3 algorithm to perform object detection

            METHODS
            =======

            Public Instance Methods:

                process_outputs(outputs, image_size): function that processes
                                                    darknet model predictions

                sigmoid(x): sigmoid function


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

    def sigmoid(self, x):
        """ sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """"
        processes darknet model predictions

        PARAMETERS
        ==========
            outputs - [list]: (grid_height, grid_width, anchor_boxes,
                                                        4 + 1 + classes)
                predictions as np.ndarrays from the Darknet model,
                    grid_height - [int]:
                        height of the grid used for the output.
                    grid_width - [int]:
                        width of the grid used for the output.
                    anchor_boxes - [int]:
                        number of anchor boxes used.
                    4 - [int]:
                        tuple containing (t_x, t_y, t_w, t_h).
                    1 - [int]:
                        box confidence.
                    classes - [int]:
                        number of classes.

            image_size - [np.ndarray]: [image_height, image_width]
                array containing the images original size

        RETURNS
        =======
            (boxes, box_confidences, box_class_probs) - [tuple]
                boxes - [list]: (grid_height, grid_width, anchor_boxes, 4)
                    processed boundary boxes for each output.
                box_confidences - [list]: (grid_height, grid_width,
                                                        anchor_boxes, 1)
                    box confidences for each output.
                box_class_probs [list]: (grid_height, grid_width,
                                                    anchor_boxes, classes)
                    box class probabilities for each output.
                """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size
        _, input_height, input_width, _ = self.model.input.shape.as_list()

        # loop through each output and its respective anchors
        for output, anchors in zip(outputs, self.anchors):

            grid_height, grid_width, num_anchors, ect = output.shape

            # empty zero arrays for each output
            b_coords = np.zeros((grid_height, grid_width, num_anchors, 4))
            b_conf = np.zeros((grid_height, grid_width, num_anchors, 1))
            c_prob = np.zeros(
                (grid_height, grid_width, num_anchors, len(
                    self.class_names)))

            for row in range(grid_width):
                for col in range(grid_height):
                    for b in range(num_anchors):

                        # raw box output coords, box confidence, and class
                        # probabilites along with the current anchor box
                        t_x, t_y, t_w, t_h = output[row, col, b, :4]
                        box_confidence = output[row, col, b, 4]
                        class_probs = output[row, col, b, 5:]
                        a_w, a_h = anchors[b]

                        # sigmoid to get a scaled value between 0 and 1
                        box_confidence = self.sigmoid(box_confidence)
                        class_probs = self.sigmoid(class_probs)

                        # find centroid and width and height of new bound box
                        b_x = (self.sigmoid(t_x) + col) / grid_width
                        b_y = (self.sigmoid(t_y) + row) / grid_height
                        b_w = (np.exp(t_w) * a_w) / input_width
                        b_h = (np.exp(t_h) * a_h) / input_height

                        # processed box coordinates
                        x1 = (b_x - (b_w / 2)) * image_width
                        y1 = (b_y - (b_h / 2)) * image_height
                        x2 = (b_x + (b_w / 2)) * image_width
                        y2 = (b_y + (b_h / 2)) * image_height

                        # update
                        b_coords[row, col, b] = [x1, y1, x2, y2]
                        b_conf[row, col, b] = box_confidence
                        c_prob[row, col, b] = class_probs

            # append each processed output to final array
            boxes.append(b_coords)
            box_confidences.append(b_conf)
            box_class_probs.append(c_prob)

        return boxes, box_confidences, box_class_probs
