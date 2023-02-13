from turtle import width
from general import xywh2xyxy
import numpy as np
import sys

def get_detection_box_yolo(detection):

    detection_copy = detection.copy()
    if len(detection_copy.shape) == 1:
        detection_copy = detection_copy.reshape(1, detection_copy.shape[0])
    
    bounding_box_output = detection_copy[:, 2:7]
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

def get_limbs_postion(detection):

    if len(detection.shape) == 1:
        detection = detection.reshape(1, detection.shape[0])

    detection_limbs = detection[:,7:]

    confidence_list_index = [i for i in range(detection_limbs.shape[1]) if (i+1) % 3 == 0 ]

    confidence_for_limb = detection_limbs[:, confidence_list_index]
    detection_limbs = np.delete(detection_limbs, confidence_list_index, axis = 1) 

    detection_limbs_human_format = detection_limbs.reshape(confidence_for_limb.shape[0], confidence_for_limb.shape[1], 2)

    return detection_limbs, confidence_for_limb, detection_limbs_human_format

def get_maximum_area_detection(detection):
    yolo_boxes = get_detection_box_yolo(detection)

    maximum = -sys.maxsize
    index_location = -1

    for idx in range(yolo_boxes.shape[0]):
       area = (yolo_boxes[idx, 2] - yolo_boxes[idx, 0]) * ((yolo_boxes[idx, 3] - yolo_boxes[idx, 1]))

       if area > maximum:
           maximum = area
           index_location = idx

    final_detection = detection[index_location]

    return final_detection