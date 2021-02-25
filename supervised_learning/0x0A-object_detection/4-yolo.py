#!/usr/bin/env python3
"""class for model Yolo"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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

    def sigmoid(self, z):
        """performs sigmoid mapping"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """Returns a tuple of (boxes, box_confidences,
           box_class_probs)"""
        boxes = []
        confidences = []
        class_proba = []
        img_H = image_size[0]
        img_W = image_size[1]
        for output in outputs:
            boxes.append(output[..., 0:4])
            confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            class_proba.append(self.sigmoid(output[..., 5:]))
        for i, box in enumerate(boxes):
            g_h, g_w, achors_box, _ = box.shape
            coordidate = np.zeros((g_h, g_w, achors_box))
            idx_y = np.arange(g_h)
            idx_y = idx_y.reshape(g_h, 1, 1)
            idx_x = np.arange(g_w)
            idx_x = idx_x.reshape(1, g_w, 1)
            C_x = coordidate + idx_x
            C_y = coordidate + idx_y
            centerX = box[..., 0]
            centerY = box[..., 1]
            width = box[..., 2]
            height = box[..., 3]
            bx = (self.sigmoid(centerX) + C_x) / g_w
            by = (self.sigmoid(centerY) + C_y) / g_h
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (np.exp(width) * pw) / self.model.input.shape[1].value
            bh = (np.exp(height) * ph) / self.model.input.shape[2].value
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            box[..., 0] = x1 * img_W
            box[..., 1] = y1 * img_H
            box[..., 2] = x2 * img_W
            box[..., 3] = y2 * img_H
        return boxes, confidences, class_proba

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Returns a tuple of (filtered_boxes, box_classes,
           box_scores)"""
        filtered_scores = []
        filtered_class = []
        filtered_boxes = []
        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            box_max_scores = np.max(box_score, axis=-1).reshape(-1)
            box_class_idx_del = np.where(box_max_scores < self.class_t)
            mask = box_max_scores >= self.class_t
            score = mask * box_max_scores
            score1 = score[score > 0]
            filtered_scores.append(score1)
            class1 = np.argmax(box_score, axis=-1).reshape(-1)
            class2 = np.delete(class1, box_class_idx_del)
            filtered_class.append(class2)
            a, b, c, _ = boxes[i].shape
            mask_reshape = mask.reshape(a, b, c, 1)
            box1 = boxes[i] * mask_reshape
            box2 = box1[box1 != 0]
            filtered_boxes.append(box2)
        filtered_scores1 = np.concatenate(filtered_scores)
        filtered_class1 = np.concatenate(filtered_class)
        filtered_boxes1 = np.concatenate(filtered_boxes)
        filtered_boxes2 = filtered_boxes1.reshape(-1, 4)
        return filtered_boxes2, filtered_class1, filtered_scores1

    def iou(self, box1, box2):
        """calculates iou"""
        x_x1 = np.maximum(box1[0], box2[0])
        y_y1 = np.maximum(box1[1], box2[1])
        x_x2 = np.minimum(box1[2], box2[2])
        y_y2 = np.minimum(box1[3], box2[3])
        inter_area = max(y_y2 - y_y1, 0) * max(x_x2 - x_x1, 0)
        box1_area = (box1[3] - box1[1])*(box1[2] - box1[0])
        box2_area = (box2[3] - box2[1])*(box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area
        iou = inter_area/union_area
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Returns a tuple of
           (box_predictions, predicted_box_classes,
            predicted_box_scores)"""
        idx = np.lexsort((-box_scores, box_classes))
        sorted_box_pred = filtered_boxes[idx]
        sorted_box_class = box_classes[idx]
        sorted_box_scores = box_scores[idx]
        _, counts = np.unique(sorted_box_class,
                              return_counts=True)
        i = 0
        n = 0
        for count in counts:
            while i < n + count:
                j = i + 1
                while j < n + count:
                    temp = self.iou(sorted_box_pred[i], sorted_box_pred[j])
                    if temp > self.nms_t:
                        sorted_box_pred = np.delete(sorted_box_pred,
                                                    j, axis=0)
                        sorted_box_scores = np.delete(sorted_box_scores,
                                                      j, axis=0)
                        sorted_box_class = np.delete(sorted_box_class,
                                                     j, axis=0)
                        count -= 1
                    else:
                        j += 1
                i += 1
            n += count
        return sorted_box_pred, sorted_box_class, sorted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Returns a tuple of (images, image_paths)
        """
        image_paths = glob.glob(folder_path + "/*")
        list_path = [cv2.imread(i) for i in image_paths]
        return list_path, image_paths
