import sys

sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\utils\\")
sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\")

import models_LSTM as models
import torch
import YoloModel as YM
import utils_detection
from torch.utils.data import DataLoader
from torch import nn
from Vizualize import *

MODEL_PATH = "F:\Licenta\Lucrare-licenta\\best_model_LSTM.pth"
VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\tennis_match_crop.mp4"
#VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\bojana_updated.mp4"
#VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\Federer1.mp4"
NAME_OF_OUTPUT = "tenis_match_crop_20_sequence"
REVERSED = True
#VIDEO_PATH = "F:\Licenta\VIDEO_RGB\\backhand_slice\\p20_bslice_s2.avi"
#VIDEO_PATH = "F:\\Licenta\\VIDEO_RGB\\backhand_volley\\p1_bvolley_s2.avi"

SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE = models.LSTM_shallow.return_train_data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.LSTM_shallow(INPUT_SIZE, hidden_units=HIDDEN_SIZE, seq_length=SEQUENCE_LENGTH).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Loaded the LSTM model")

model_YOLO = YM.YoloModel()
model_YOLO.inference = True
model_YOLO.training_mode = True

print("Loaded the YOLO model")

all_detection, yolo_boxes = model_YOLO.read_from_video(VIDEO_PATH)
print("Read video file")

all_detections_normalized = utils_detection.normalize_detection_limbs(yolo_boxes, all_detection)

all_detections_normalized = utils_detection.convert_to_2D_matrix(all_detections_normalized)
all_detections_sequence, _ = utils_detection.get_all_sequences_from_2D_format(all_detections_normalized, SEQUENCE_LENGTH, 0, SEQUENCE_LENGTH)
all_detections_sequence = torch.tensor(all_detections_sequence).float().to(device)

output_labels = []
confidence = []

test_loader = DataLoader(all_detections_sequence, 1, shuffle=False)

with torch.no_grad():
    for X in test_loader:
        output = model(X)
        softmax = nn.Softmax(dim = 1)
        output = softmax(output)
        maximum_values, predicted = torch.max(output.data, 1)
        for i in range(len(predicted)):
            if maximum_values[i] < 0.7:
                predicted[i] = -1
        
        output_labels.append(predicted[i].item())
        confidence.append(maximum_values[i].item())

#print("TOT OUTPUT",output_labels)
output_names = [None for i in range(len(output_labels))]

for i in range(len(output_labels)):
    if REVERSED == True:
        output_names[i] = decode_output(output_labels[i])[2]
    else:
        output_names[i] = decode_output(output_labels[i])[0]

video_write(VIDEO_PATH, yolo_boxes, output_names, confidence, NAME_OF_OUTPUT)
