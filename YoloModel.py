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

class YoloModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load('yolov7\yolov7-w6-pose.pt', map_location=self.device)
        self.model = weigths['model']
        _ = self.model.float().eval()
        
        self.yolo_boxes = None
        self.inference = False
        self.training_mode = False
        self.verbose = False

        if torch.cuda.is_available():
            self.model.half().to(self.device)

    @torch.no_grad()
    def inference_on_image(self, image):
        #x = time.perf_counter()
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(self.device)   

        output, _ = self.model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        #y = time.perf_counter()
        #print(y-x)

        return output, image_

    def read_from_video(self, video_path):
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

        #capWriter = cv2.VideoWriter(f"keypoint.mp4",
        #                        cv2.VideoWriter_fourcc(*'mp4v'), fps,
        #                        (resize_width, resize_height))

        all_detections = None
        yolo_boxes = None

        while cap.isOpened():
            ret, frame = cap.read()

            if ret == True:
                image = frame

                output, image_ = self.inference_on_image(image)
                image_drawn = image_

                if output.shape[0] != 0:
                    
                    if self.training_mode == True:
                        output = get_maximum_area_detection(output)

                    detection_boxes = get_detection_box_yolo(output)
                    detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)

                    if all_detections is None:
                        all_detections = detection_limbs_human_format
                        yolo_boxes = detection_boxes
                    else:
                        all_detections = np.append(all_detections, detection_limbs_human_format, axis = 0)
                        yolo_boxes = np.append(yolo_boxes, detection_boxes, axis = 0)

                    #image_drawn = visualize_box_detection(detection_boxes, output, image_)
                
                elif self.inference == True:
                    #if no box was detected then we will return the invalid values being full -1
                    if yolo_boxes is None:
                        yolo_boxes = np.array([-1, -1, -1, -1, -1])
                        all_detections = np.array([[0 for i in range(34)]])
                    else:
                        yolo_boxes = np.append(yolo_boxes, np.array([[-1, -1, -1, -1, -1]]), axis = 0)
                        all_detections = np.append(all_detections, np.array([[[0, 0] for i in range(17)]]), axis = 0)

                #capWriter.write(image_drawn)

            else:
                break

        self.yolo_boxes = yolo_boxes
        cap.release()
        #capWriter.release()

        return all_detections, yolo_boxes

    def capture_from_camera(self):
        # define a video capture object
        vid = cv2.VideoCapture(0)
    
        while(True):
        
            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            image = frame
            output, image_ = self.inference_on_image(image)
            image_drawn = image_

            if output.shape[0] != 0:

                detection_boxes = get_detection_box_yolo(output)
                detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
                detection_normalized = normalize_detection_limbs(detection_boxes, detection_limbs_human_format)
                print(detection_normalized)
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