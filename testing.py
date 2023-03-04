import sys

sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\utils\\")
sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\")

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
import copy
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

BATCH_SIZE = 64
SEQUENCE_LENGTH = 16

torch.manual_seed(99)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_file_names = DatasetLoader.load_test_from_file("testing_files.txt") 
DL_training = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, False, test_file_names)
DL_test = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, True, test_file_names)

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
        self.num_layers = 3

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.6)
        
        self.fc_1  = nn.Linear(self.hidden_units, 256)
        self.fc_final = nn.Linear(256, self.num_classes) 
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to("cuda:0").requires_grad_() #.to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to("cuda:0").requires_grad_() #.to(device) inainte de grad

        outputs, (hn, _) = self.lstm(x, (h0, c0))
        outputs = outputs[:, -1, :]
        outputs = self.dropout(outputs)
        out = self.relu(outputs)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_final(out)

        return out

num_epochs = 40
learning_rate = 1e-4 #cu 1e-4 converge mai rpd

input_size = 34
hidden_size = 512
model = LSTM(input_size, hidden_units=hidden_size, seq_length=SEQUENCE_LENGTH).to("cuda:0")#.to(device)
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def test_model(test_loader, model, loss_function):
    num_batches = len(test_loader)
    total_loss = 0

    model.eval()

    specifics = [0 for i in range(12)] #variable that holds correct guessed value

    with torch.no_grad():
        nr_true = 0
        nr_false = 0
        metric2_true = 0
        metric2_false = 0
        for X,y in iter(test_loader):
            output = model(X)
            total_loss += loss_function(output, y).item()
            #print(output.data)
            softmax = nn.Softmax(dim = 1)
            output = softmax(output)
            maximum_values, predicted = torch.max(output.data, 1)
            
            for i in range(len(predicted)):
                if maximum_values[i] < 0.5:
                    predicted[i] = -1

            #print(output)
            #print("predicted=", predicted)
            #print("y=", y)
            for i, element in enumerate(predicted==y):
                if element == True:
                    nr_true += 1
                    specifics[predicted[i]] += 1
                else:
                    nr_false += 1

            #for i in range(len(y)):
            #    if y[i] in [0,1,2,3]:
            #        y[i] = 0
            #    elif y[i] in [4,5,6,7,8]:
            #        y[i] = 1
            #    elif y[i] in [9,10]:
            #        y[i] = 2
            #    elif y[i] in [11]:
            #        y[i] = 3
#
            #for i in range(len(predicted)):
            #    if predicted[i] in [0,1,2,3]:
            #        predicted[i] = 0
            #    elif predicted[i] in [4,5,6,7,8]:
            #        predicted[i] = 1
            #    elif predicted[i] in [9,10]:
            #        predicted[i] = 2
            #    elif predicted[i] in [11]:
            #        predicted[i] = 3
#
            #for element in (predicted==y):
            #    if element == True:
            #        metric2_true += 1
            #    else:
            #        metric2_false += 1

    print(f'    Metric 1, num true = {nr_true}, num_false = {nr_false}, precizie = {nr_true / (nr_true + nr_false)}')
    #print(f'Metric 2, num true = {metric2_true}, num_false = {metric2_false}, precizie = {metric2_true / (metric2_true + metric2_false)}')
    avg_loss = total_loss / num_batches
    print(f"    Test loss: {avg_loss}")
    return avg_loss

num_batches = len(train_loader)

epochs = []
train_loss = []
validation_loss = []
minimum_testing_error = 99999
minimum_epoch = None
minimum_model = None

for epoch in range(num_epochs):
    time_begin = time.perf_counter()
    model.train()
    total_loss = 0
    epochs.append(epoch)
    for X, y in train_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3) #model mare, 0.8
        optimizer.step()

        total_loss += loss.item()

    time_end = time.perf_counter()

    avg_loss = total_loss / num_batches
    print(f"Train loss for epoch {epoch}: {avg_loss}, duration {time_end-time_begin} seconds")
    train_loss.append(avg_loss)
    test_loss = test_model(test_loader, model, loss_function)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"intermediary_results\\saved_checkpoint_{model.__class__.__name__}_{epoch}_epoch.pth")

    if test_loss < minimum_testing_error:
        minimum_testing_error = test_loss
        minimum_model = copy.deepcopy(model.state_dict())
        minimum_epoch = epoch
        

    validation_loss.append(test_loss)
    
torch.save(model.state_dict(), "saved.pth")
print(f"Minimum epoch {minimum_epoch}, minimum loss {minimum_testing_error}")
torch.save(minimum_model, f"best_model_{model.__class__.__name__}.pth")


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