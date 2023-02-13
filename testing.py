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
from torch_utils import *
from Vizualize import *

import time

def initialize_model(device):
    weigths = torch.load('yolov7\yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    return model

@torch.no_grad()
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


def read_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    width, height, fps = None, None, None

    if cap.isOpened() == False:
        print("Error in opening file")
        return
    else:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))

    vid_write_image = letterbox(cap.read()[1], 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]

    capWriter = cv2.VideoWriter(f"keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (resize_width, resize_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            image = frame

            output, image_ = inference_on_image(image)

            output = get_maximum_area_detection(output)
            detection_boxes = get_detection_box_yolo(output)

            detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
            image_drawn = visualize_box_detection(detection_boxes, output, image_)
            capWriter.write(image_drawn)
        else:
            break

    cap.release()
    capWriter.release()


def capture_from_camera():
    # define a video capture object
    vid = cv2.VideoCapture(0)
  
    while(True):
      
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        image = frame

        output, image_ = inference_on_image(image)


        detection_boxes = get_detection_box_yolo(output)

        detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
        image_drawn = visualize_box_detection(detection_boxes, output, image_)

        # Display the resulting frame
        cv2.imshow('frame', image_drawn)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = initialize_model(device)
#read_from_video("C:\\Licenta\\VIDEO_RGB\\backhand2hands\\p4_backhand2h_s1.avi")
capture_from_camera()
#image = cv2.imread('C:\Licenta\Lucrare-licenta\yolov7\inference\images\zidane.jpg')
#
#output, image_ = inference_on_image(image)
#
#output = get_maximum_area_detection(output)
#detection_boxes = get_detection_box_yolo(output)
#
#detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
#print(detection_limbs_human_format)
#image_drawn = visualize_box_detection(detection_boxes, output, image_)
#cv2.imshow("LALA", image_drawn)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

