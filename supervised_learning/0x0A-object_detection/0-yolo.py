#!/usr/bin/env python3
"""class for model Yolo"""
import tensorflow.keras as K


class Yolo():
    """class Yolo that uses Yolov3 algorithm"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """constructor for a Yolo class"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            lines = [line.split("\n")[0] for line in classes.readlines()]
        self.class_names = lines
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
