from general import xywh2xyxy
import numpy as np
import sys
import pandas as pd

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

    confidence_for_limb = confidence_for_limb < 0.5
    
    for idx in range(confidence_for_limb.shape[0]):
        for jdx in range(confidence_for_limb.shape[1]):
            if confidence_for_limb[idx,jdx] == True:
                detection_limbs_human_format[idx,jdx,0] = 0
                detection_limbs_human_format[idx,jdx,1] = 0


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

def save_to_csv_limbs(csv_name, limbs_location_human_format):
    format = convert_to_2D_matrix(limbs_location_human_format)

    df = pd.DataFrame(format)
    df.to_csv(csv_name)

def load_from_csv_limbs(csv_name):
    df_numpy = pd.read_csv(csv_name).to_numpy()[:, 1:]
    return df_numpy

def save_to_csv_YOLO(csv_name, yolo_boxes):
    df = pd.DataFrame(yolo_boxes)
    df.to_csv(csv_name)

def load_from_csv_YOLO(yolo_boxes):
    df_numpy = pd.read_csv(yolo_boxes).to_numpy()[:, 1:]
    return df_numpy

def convert_to_2D_matrix(detection_limbs_human_format):
    all_detections_2D_matrix = detection_limbs_human_format.reshape(detection_limbs_human_format.shape[0], detection_limbs_human_format.shape[1] * detection_limbs_human_format.shape[2])
    return all_detections_2D_matrix

def convert_to_csv_format_normalized(detection_limbs_human_format, yolo_boxes):
    all_detections = normalize_detection_limbs(yolo_boxes, detection_limbs_human_format)
    all_detections = convert_to_2D_matrix(all_detections)
    return all_detections

def normalize_detection_limbs(yolo_boxes, detection_limbs):
    #normalizez by the same yolo_box corresponding to the detection
    #detection limbs is in human_format
    detection_for_limbs = detection_limbs.copy()

    for idx in range(yolo_boxes.shape[0]):
        x_std_normalization = yolo_boxes[idx, 2] - yolo_boxes[idx, 0]
        y_std_normalization = yolo_boxes[idx, 3] - yolo_boxes[idx, 1]

        detection_for_limbs[idx, :, 0] = (detection_for_limbs[idx, :, 0] - yolo_boxes[idx, 0]) / x_std_normalization
        detection_for_limbs[idx, :, 1] = (detection_for_limbs[idx, :, 1] - yolo_boxes[idx, 1]) / y_std_normalization

    for idx in range(detection_for_limbs.shape[0]):
        for jdx in range(detection_for_limbs.shape[1]):
            if detection_for_limbs[idx,jdx,0] < 0 or detection_for_limbs[idx,jdx,1] < 0:
                detection_for_limbs[idx,jdx,0] = 0
                detection_for_limbs[idx,jdx,1] = 0

    return detection_for_limbs

def convert_2D_Human(matrix):
    return matrix.reshape(matrix.shape[0], matrix.shape[1] // 2, 2)


def get_all_sequences_from_2D_format(matrix, sequence_length, y_value, step = 1):

    no_rows = matrix.shape[0]

    sequences = []

    for index_rows in range(0, no_rows - sequence_length + 1, step):
        sequences.append(matrix[index_rows : index_rows + sequence_length])

    sequences = np.array(sequences)
    y = np.full(sequences.shape[0], y_value)
    
    return sequences, y
        