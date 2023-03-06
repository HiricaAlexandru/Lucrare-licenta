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
import models_LSTM

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

#loading the data necessary for model creation
BATCH_SIZE = 512
SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE = models_LSTM.LSTM_shallow_23_sequence.return_train_data()
MODEL_PATH = "F:\Licenta\Lucrare-licenta\models\LSTM_shallow_23_sequence\saved_checkpoint_LSTM_27_epoch_best.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_file_names = DatasetLoader.load_test_from_file("testing_files.txt") 
DL_test = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, True, test_file_names)

test_loader = DataLoader(DL_test, BATCH_SIZE, shuffle=False)


model = models_LSTM.LSTM_shallow_23_sequence(INPUT_SIZE, hidden_units=HIDDEN_SIZE, seq_length=SEQUENCE_LENGTH).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

specifics = [0 for i in range(12)] #variable that holds correct guessed value

with torch.no_grad():
        nr_true = 0
        nr_false = 0
        metric2_true = 0
        metric2_false = 0
        for X,y in iter(test_loader):
            output = model(X)
            softmax = nn.Softmax(dim = 1)
            output = softmax(output)
            maximum_values, predicted = torch.max(output.data, 1)
            
            for i in range(len(predicted)):
                if maximum_values[i] < 0.5:
                    predicted[i] = -1
            for i, element in enumerate(predicted==y):
                if element == True:
                    nr_true += 1
                    specifics[predicted[i]] += 1
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

loss_function = nn.CrossEntropyLoss()
print("Eroare entropie = ", test_model(test_loader, model, loss_function))

print(f'Metric 1, num true = {nr_true}, num_false = {nr_false}, precision = {nr_true / (nr_true + nr_false)}')
print(f'Metric 2, num true = {metric2_true}, num_false = {metric2_false}, precision = {metric2_true / (metric2_true + metric2_false)}')
print(specifics)