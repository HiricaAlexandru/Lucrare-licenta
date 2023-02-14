import sys

sys.path.append("C:\\Licenta\\Lucrare-licenta\\yolov7\\utils\\")
sys.path.append("C:\\Licenta\\Lucrare-licenta\\yolov7\\")

import YoloModel as YM
from utils_detection import *
import time
import pandas as pd

model = YM.YoloModel()
model.training_mode = True
#model.capture_from_camera()
all_detection, yolo_boxes = model.read_from_video("C:\\Licenta\\VIDEO_RGB\\backhand\\p29_backhand_s1.avi")
all_dec_mat = convert_to_2D_matrix(all_detection)
save_to_csv_limbs("nume.csv", all_detection)
all_detection_loaded = load_from_csv_limbs("nume.csv")

save_to_csv_YOLO("yolo_boxes.csv", yolo_boxes)
yolo_b = load_from_csv_YOLO("yolo_boxes.csv")

all_detections_made_algo = load_from_csv_limbs("C:\Licenta\Dataset\\backhand\p29_backhand_s1_limbs.csv")
yolo_ba = load_from_csv_YOLO("C:\Licenta\Dataset\\backhand\p29_backhand_s1_yolo.csv")

print(all_detections_made_algo == all_dec_mat)
print(yolo_boxes == yolo_ba)


##model_parameters = filter(lambda p: p.requires_grad, model.parameters())
##params = sum([np.prod(p.size()) for p in model_parameters])
##print(params)
##read_from_video("C:\\Licenta\\VIDEO_RGB\\backhand2hands\\p4_backhand2h_s1.avi")
#capture_from_camera()
#image = cv2.imread('C:\Licenta\Lucrare-licenta\yolov7\inference\images\zidane.jpg')
#
#output, image_ = inference_on_image(image)
#
##output = get_maximum_area_detection(output)
#detection_boxes = get_detection_box_yolo(output)
##
#detection_limbs, confidence_for_limb, detection_limbs_human_format = get_limbs_postion(output)
#detection_normalized = normalize_detection_limbs(detection_boxes, detection_limbs_human_format)
#print(detection_normalized)
#image_drawn = visualize_box_detection(detection_boxes, output, image_)
#cv2.imshow("LALA", image_drawn)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#