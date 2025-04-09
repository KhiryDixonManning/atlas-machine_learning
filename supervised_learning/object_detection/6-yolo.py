#!/usr/bin/env python3
""" module containing class that represents the YOLOv3 algorithm """
import os
import numpy as np
import tensorflow as tf
import cv2
import os


class Yolo():
    """ class that represents the YOLOv3 algorithm to perform object detection

            METHODS
            =======

            Public Instance Methods:

                process_outputs(outputs, image_size): function that processes
                                                    darknet model predictions

                sigmoid(x): sigmoid function

                filter_boxes(boxes, box_confidences, box_class_probs): Filter
                    the bounding boxes based on box confidences and class
                    probabilities.

                non_max_suppression(filtered_boxes, box_classes, box_scores):
                    performs non-max suppression on the bounding boxes
                    predictions

                find_iou(box1, box2):
                    calculates the Intersection over Union (IoU)
                    between two bounding boxes

                preprocess_images(images):
                    preprocesses images for input to the YOLO model

                show_boxes(image, boxes, box_classes, box_scores, file_name):
                    displays the image with boundary boxes, class names,
                    and box scores.

            Static Methods:

                load_images(folder_path):
                    loads all images from a folder



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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        filters the bounding boxes based on box confidences and class
        probabilities.

        PARAMETERS
        ==========
            boxes - [list]: (grid_height, grid_width, anchor_boxes, 4)
                processed boundary boxes for each output as np.ndarrays

            box_confidences - [list]: (grid_height, grid_width,
                                                    anchor_boxes, 1)
                processed box confidences for each output as np.ndarrays

            box_class_probs - [list]: (grid_height, grid_width,
                                                    anchor_boxes, classes)
                processed box class probabilities for each output as np.arrays

        RETURNS
        =======
            (filtered_boxes, box_classes, box_scores) - [tuple]:
                filtered_boxes - [np.ndarray]: (?, 4)
                    Filtered bounding boxes.
                        ?: Number of filtered boxes
                box_classes - [np.ndarray]: (?,)
                    class number that each box in filtered_boxes predicts.
                        ?: Number of filtered boxes
                box_scores - [np.ndarray]: (?)
                    box scores for each box in filtered_boxes.
                        ?: Number of filtered boxes
        """
        # Flatten the lists of boxes, confidences, and class probabilities
        box_scores = []
        box_classes = []
        filtered_boxes = []

        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                for x in range(len(boxes[i][j])):
                    for y in range(len(boxes[i][j][x])):
                        max_class_prob = np.max(box_class_probs[i][j][x][y])
                        box_score = box_confidences[i][j][x][y][0] * \
                            max_class_prob
                        if box_score > self.class_t:
                            box_scores.append(box_score)
                            box_classes.append(
                                list(box_class_probs[i][j][x][y]).index(
                                    max_class_prob))
                            filtered_boxes.append(boxes[i][j][x][y])

        return np.array(filtered_boxes), np.array(
            box_classes), np.array(box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        performs non-max suppression on the bounding boxes predictions

        PARAMETERS
        ==========
            filtered_boxes - [numpy.ndarray]: (?, 4)
                containins all of the filtered bounding boxes
            box_classes - [numpy.ndarray]: (?,)
                A numpy.ndarray of shape (?,) containing the class number
                for the class that filtered_boxes predicts, respectively
            box_scores - [numpy.ndarray]: (?)
                contains the box scores for each box in filtered_boxes,
                respectively

        RETURNS
        =======
            (nms_box, nms_box_classes, nms_box_scores) - [tuple]:
                nms_box - [numpy.ndarray]: (?, 4)
                    contains all of the predicted bounding boxes ordered by
                    class and box score
                nms_box_classes - [numpy.ndarray]: (?,)
                    contains the class number for nms_box ordered by class
                    and box score, respectively
                predicted_box_scores - [numpy.ndarray]: (?)
                    containing the box scores for nms_box ordered by class
                    and box score, respectively
        """
        nms_box = []
        nms_box_classes = []
        nms_box_scores = []
        # sorting
        all_classes = {}
        for i in range(len(self.class_names)):
            one_class = []
            for j in range(len(filtered_boxes)):
                if box_classes[j] == i:
                    one_box = {}
                    one_box["score"] = box_scores[j]
                    one_box["box"] = filtered_boxes[j]
                    one_class.append(one_box)
            all_classes[i] = one_class
        sorted_classes = {}
        for x in range(len(self.class_names)):
            sorted_data = sorted(
                all_classes[x],
                key=lambda x: x['score'],
                reverse=True)
            sorted_classes[x] = sorted_data
        # sorted by class and conf

        final_list = []
        for i in range(len(sorted_classes)):
            for j in range(len(sorted_classes[i])):
                if len(sorted_classes[i][j]) > 0:
                    final_list.append((i, sorted_classes[i][j]))
                    for k in range(j + 1, len(sorted_classes[i])):
                        if len(sorted_classes[i][k]) > 0:
                            iou = self.find_iou(
                                sorted_classes[i][j]["box"],
                                sorted_classes[i][k]["box"])
                            if iou > self.nms_t:
                                sorted_classes[i][k] = {}
        for i in range(len(final_list)):
            nms_box.append(final_list[i][1]['box'])
            nms_box_classes.append(final_list[i][0])
            nms_box_scores.append(final_list[i][1]['score'])

        return np.array(nms_box), np.array(
            nms_box_classes), np.array(nms_box_scores)

    def find_iou(self, box1, box2):
        """
        calculates the Intersection over Union (IoU) between two bounding
        boxes

        PARAMETERS
        ==========
            box1 - [list]: (x1, y1, x2, y2)
                (x1, y1):
                    coordinates of top-left corner
                (x2, y2)
                    coordinates of bottom-right corner

            box2 - [list]: (x1, y1, x2, y2)
                (x1, y1):
                    coordinates of top-left corner
                (x2, y2)
                    coordinates of bottom-right corner

        RETURNS
        =======
            iou - [float]
                IoU score between the two bounding boxes
        """
        a1, b1, c1, d1 = box1
        a2, b2, c2, d2 = box2

        a_inter = max(a1, a2)
        b_inter = max(b1, b2)
        c_inter = min(c1, c2)
        d_inter = min(d1, d2)

        if c_inter < a_inter or d_inter < b_inter:
            return 0

        intersection = (c_inter - a_inter) * (d_inter - b_inter)
        b1_area = (c1 - a1) * (d1 - b1)
        b2_area = (c2 - a2) * (d2 - b2)
        union = b1_area + b2_area - intersection

        iou = intersection / float(union)
        return iou

    @staticmethod
    def load_images(folder_path):
        """
        loads all images from a folder

        PARAMETERS
        ==========
            folder_path - [str]
                the path to the folder holding all the images to load

        RETURNS
        =======
            (images, image_paths) - [tuple]
                images - [list]
                    all images as numpy.ndarrays
                image_paths - [list]
                    every path to the individual images in images
        """
        imgs = []

        # load image paths
        image_paths = tf.io.gfile.glob(folder_path + '/*')

        # loop through each path
        for file_path in image_paths:
            img = tf.io.read_file(file_path)
            # decode
            img = tf.image.decode_image(img, channels=3)
            # change from rgb to bgr
            img = img[:, :, ::-1]
            imgs.append(np.array(img))

        return imgs, image_paths

    def preprocess_images(self, images):
        """
        preprocesses images for input to the YOLO model

        PARAMETERS
        ==========
            images - [list]:
                all images as numpy.ndarrays

        RETURNS
        =======
            (pimages, image_shapes) - [tuple]:
                pimages - [numpy.ndarray]: (ni, input_h, input_w, 3)
                    contains all of the preprocessed images
                        ni - [int]:
                            the number of images that were preprocessed,
                        input_h - [int]:
                            input height for the Darknet model
                        input_w - [int]:
                            input width for the Darknet model
                        3:
                            the number of color channels.

                image_shapes - [numpy.ndarray]: (ni, 2)
                    contains the original height and width of the images
        """
        resized = []
        og_shape = []
        _, h, w, _ = self.model.input.shape
        for image in images:
            og_shape.append((image.shape[0], image.shape[1]))
            resize = tf.image.resize(image, (h, w), method='bicubic')
            minimum = np.min(resize)
            maximum = np.max(resize)
            rescaled_image = (resize - minimum) / (maximum - minimum)
            resized.append(rescaled_image)
        return np.array(resized), np.array(og_shape)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        displays the image with boundary boxes, class names, and box scores.

        PARAMETERS
        ==========
            image - [numpy.ndarray]:
                unprocessed image as a numpy.ndarray

            boxes - [numpy.ndarray]:
                the boundary boxes for the image

            box_classes - [numpy.ndarray]:
                the class indices for each box

            box_scores - [numpy.ndarray]:
                the box scores for each box

            file_name - [str]:
                file path where the original image is stored

        RETURNS
        =======
            None
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            im = cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), color=(
                    255, 0, 0), thickness=2)
            bc = box_classes[i]
            bs = box_scores[i]
            im = cv2.putText(
                im,
                f'{self.class_names[bc]} {str(round(bs, 2))}',
                (int(x1),
                 int(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(
                    0,
                    0,
                    255),
                thickness=1)
        cv2.imshow(file_name, im)
        key_press = cv2.waitKey(0)
        if key_press == ord('s'):
            if os.path.exists('detections') is False:
                os.makedirs('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, im)
            os.chdir('..')
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()
