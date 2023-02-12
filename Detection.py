import sys

sys.path.append("D:\\licenta\\yolov7\\utils\\")
sys.path.append("D:\\licenta\\yolov7\\")

print(sys.path)
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from datasets import letterbox
from general import non_max_suppression_kpt, xywh2xyxy
from plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

def get_detection_box_yolo(detection, image):
    bounding_box_output = detection[:, 2:7]
    bounding_box_output = xywh2xyxy(bounding_box_output)
    for idx in range(bounding_box_output.shape[0]):
        print(bounding_box_output[idx])
        plot_one_box(bounding_box_output[idx], image)

    cv2.imshow("PULA",image)
    cv2.waitKey(0)
  
    # closing all open windows
    cv2.destroyAllWindows()
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7\yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

image = cv2.imread('D:\licenta\yolov7\inference\images\zidane.jpg')
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
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

get_detection_box_yolo(output, image_)
