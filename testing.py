import sys

sys.path.append("D:\\Lucrare-licenta\\yolov7\\utils\\")
sys.path.append("D:\\Lucrare-licenta\\yolov7\\")

import torch
import YoloModel as YM
from utils_detection import *
import time
import pandas as pd
from DatasetLoader import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter

#model = YM.YoloModel()
#model.training_mode = True
#model.capture_from_camera()
#all_detection, yolo_boxes = model.read_from_video("C:\\Licenta\\VIDEO_RGB\\backhand\\p29_backhand_s1.avi")
#all_dec_mat = convert_to_2D_matrix(all_detection)
#save_to_csv_limbs("nume.csv", all_detection)
#all_detection_loaded = load_from_csv_limbs("nume.csv")

#save_to_csv_YOLO("yolo_boxes.csv", yolo_boxes)
#yolo_b = load_from_csv_YOLO("yolo_boxes.csv")
#x = time.perf_counter()
#all_detections_made_algo = load_from_csv_limbs("C:\Licenta\Dataset\\backhand\p29_backhand_s1_limbs.csv")
#yolo_ba = load_from_csv_YOLO("C:\Licenta\Dataset\\backhand\p29_backhand_s1_yolo.csv")
#
#all_detections_made_algo2 = load_from_csv_limbs("C:\Licenta\Dataset\\backhand\p29_backhand_s2_limbs.csv")
#yolo_ba2 = load_from_csv_YOLO("C:\Licenta\Dataset\\backhand\p29_backhand_s2_yolo.csv")
#
##print(all_detections_made_algo)
#matrix_inverse = convert_2D_Human(all_detections_made_algo)
#print(matrix_inverse.shape)
#
#matrice_norm = normalize_detection_limbs(yolo_ba, matrix_inverse)
#csv_format = convert_to_2D_matrix(matrice_norm)
#sequences, y_first = get_all_sequences_from_2D_format(csv_format, 3, 1) #trebuie cea normalizata
#
#y = time.perf_counter()
#print(y-x)
#
#matrix_inverse = convert_2D_Human(all_detections_made_algo2)
#
#matrice_norm = normalize_detection_limbs(yolo_ba2, matrix_inverse)
#csv_format = convert_to_2D_matrix(matrice_norm)
#sequences2, y_second = get_all_sequences_from_2D_format(csv_format, 3, 2) #trebuie cea normalizata
#print(sequences.shape, sequences2.shape)
#print(np.append(sequences, sequences2, axis = 0).shape)
#print(y_first.shape, y_second.shape)
#print(np.append(y_first, y_second))

#useful!

from torch.utils.data import DataLoader

BATCH_SIZE = 16
SEQUENCE_LENGTH = 10

torch.manual_seed(99)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_file_names = DatasetLoader.load_test_from_file("testing_files.txt") 
DL_training = DatasetLoader("D:\Lucrare-licenta\Dataset", SEQUENCE_LENGTH, device, False, test_file_names)
DL_test = DatasetLoader("D:\Lucrare-licenta\Dataset", SEQUENCE_LENGTH, device, True, test_file_names)

train_loader = DataLoader(DL_training, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(DL_test, BATCH_SIZE, shuffle=False)


from torch import nn

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True)
        
        self.fc_1  = nn.Linear(self.hidden_units, 128)
        self.fc_final = nn.Linear(128, self.num_classes) 

        self.relu = nn.ReLU()
        

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_() #.to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_() #.to(device) inainte de grad

        _, (hn, _) = self.lstm(x, (h0, c0))
        hn = hn.view(-1, self.hidden_units)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_final(out)

        return out

num_epochs = 10
learning_rate = 0.0001

input_size = 34
hidden_size = 64
model = LSTM(input_size, hidden_units=hidden_size, seq_length=SEQUENCE_LENGTH) #.to(device)
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



def test_model(test_loader, model, loss_function):
    num_batches = len(test_loader)
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss


num_batches = len(train_loader)

epochs = []
train_loss = []
validation_loss = []


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    epochs.append(epoch)
    for X, y in train_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    #with torch.no_grad():
    #    nr_true = 0
    #    nr_false = 0
    #    for X,y in iter(test_loader):
    #        output = model(X)
    #        _, predicted = torch.max(output.data, 1)
    #        #print(output)
    #        #print(predicted)
    #        for element in (predicted==y):
    #            if element == True:
    #                nr_true += 1
    #            else:
    #                nr_false += 1
#
    #    print(f"nr true: {nr_true}, nr false {nr_false}")
#

    avg_loss = total_loss / num_batches
    print(f"Train loss for epoch {epoch}: {avg_loss}")
    train_loss.append(avg_loss)
    validation_loss.append(test_model(test_loader, model, loss_function))
    
torch.save(model.state_dict(), "saved.pth")

plt.plot(epochs, train_loss)
plt.plot(epochs, validation_loss)
plt.show()
#X, y = next(iter(DL))
#print(X.shape)
#print(y)


#print(all_detections_made_algo == all_dec_mat)
#print(yolo_boxes == yolo_ba)


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