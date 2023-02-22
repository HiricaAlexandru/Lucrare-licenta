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
from torch.utils.data import DataLoader
from torch import nn

BATCH_SIZE = 512
SEQUENCE_LENGTH = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_file_names = DatasetLoader.load_test_from_file("testing_files.txt") 
DL_test = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, True, test_file_names)

test_loader = DataLoader(DL_test, BATCH_SIZE, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 10

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.2)
        
        self.fc_1  = nn.Linear(self.hidden_units, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_final = nn.Linear(128, self.num_classes) 

        self.dropout = nn.Dropout(0.5)
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
        out = self.fc_2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_final(out)

        return out

input_size = 34
hidden_size = 40
model = LSTM(input_size, hidden_units=hidden_size, seq_length=SEQUENCE_LENGTH).to("cuda:0")#.to(device)
model.load_state_dict(torch.load("F:\Licenta\Lucrare-licenta\\best_model_16.pth"))
model.eval()

with torch.no_grad():
        nr_true = 0
        nr_false = 0
        metric2_true = 0
        metric2_false = 0
        for X,y in iter(test_loader):
            output = model(X)
            #print(output.data)
            softmax = nn.Softmax(dim = 1)
            output = softmax(output)
            maximum_values, predicted = torch.max(output.data, 1)
            
            for i in range(len(predicted)):
                if maximum_values[i] < 0.4:
                    predicted[i] = -1

            #print(output)
            #print("predicted=", predicted)
            #print("y=", y)
            for element in (predicted==y):
                if element == True:
                    nr_true += 1
                else:
                    nr_false += 1

            for i in range(len(y)):
                if y[i] in [0,1,2,3]:
                    y[i] = 0
                elif y[i] in [4,5,6,7,8]:
                    y[i] = 1
                elif y[i] in [9,10]:
                    y[i] = 2
                elif y[i] in [11]:
                    y[i] = 3

            for i in range(len(predicted)):
                if predicted[i] in [0,1,2,3]:
                    predicted[i] = 0
                elif predicted[i] in [4,5,6,7,8]:
                    predicted[i] = 1
                elif predicted[i] in [9,10]:
                    predicted[i] = 2
                elif predicted[i] in [11]:
                    predicted[i] = 3

            for element in (predicted==y):
                if element == True:
                    metric2_true += 1
                else:
                    metric2_false += 1

print(f'Metric 1, num true = {nr_true}, num_false = {nr_false}')
print(f'Metric 2, num true = {metric2_true}, num_false = {metric2_false}')