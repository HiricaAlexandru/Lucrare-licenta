import sys

sys.path.append("C:\\Licenta\\Lucrare-licenta\\yolov7\\utils\\")
sys.path.append("C:\\Licenta\\Lucrare-licenta\\yolov7\\")

import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from datasets import letterbox
from general import non_max_suppression_kpt
from plots import output_to_keypoint, plot_skeleton_kpts
from utils_detection import *
from Vizualize import *

import time

def initialize_model(device):
    weigths = torch.load('yolov7\yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    return model

def inference_on_image(image):
    x = time.perf_counter()
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)   
    output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)
    y = time.perf_counter()
    print(y-x)

    return output, image_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = initialize_model(device)
image = cv2.imread('C:\Licenta\Lucrare-licenta\yolov7\inference\images\zidane.jpg')

output, image_ = inference_on_image(image)

output = get_maximum_area_detection(output)
detection_boxes = get_detection_box_yolo(output)

detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
print(detection_limbs_human_format)
image_drawn = visualize_box_detection(detection_boxes, output, image_)
cv2.imshow("LALA", image_drawn)
cv2.waitKey(0)
cv2.destroyAllWindows()

