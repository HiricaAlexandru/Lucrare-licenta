from general import xywh2xyxy
import numpy as np

def get_detection_box_yolo(detection):
    bounding_box_output = detection[:, 2:7]
    bounding_box_output = xywh2xyxy(bounding_box_output)
    return bounding_box_output

def resize_boxes_detection(detection_boxes, original_size, target_size):
    resize_factor_x = target_size[0] / original_size[0]
    resize_factor_y = target_size[1] / original_size[1]

    detection_boxes[:, 0] = np.round(detection_boxes[:, 0] * resize_factor_y)
    detection_boxes[:, 2] = np.round(detection_boxes[:, 2] * resize_factor_y)
    detection_boxes[:, 1] = np.round(detection_boxes[:, 1] * resize_factor_x)
    detection_boxes[:, 3] = np.round(detection_boxes[:, 3] * resize_factor_x)

    return detection_boxes


