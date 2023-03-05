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

BATCH_SIZE = 128
SEQUENCE_LENGTH = 20

torch.manual_seed(99)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_file_names = DatasetLoader.load_test_from_file("testing_files.txt") 
DL_training = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, False, test_file_names)
DL_test = DatasetLoader("F:\Licenta\Dataset", SEQUENCE_LENGTH, device, True, test_file_names)

train_loader = DataLoader(DL_training, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(DL_test, BATCH_SIZE, shuffle=False)


from torch import nn

class LSTM(nn.Module):
    #best results!
    #batch size 64
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
    
    def return_train_data():
        SEQUENCE_LENGTH = 16
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE

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

            for i, element in enumerate(predicted==y):
                if element == True:
                    nr_true += 1
                    specifics[predicted[i]] += 1
                else:
                    nr_false += 1

    print(f'    Metric 1, num true = {nr_true}, num_false = {nr_false}, precision = {nr_true / (nr_true + nr_false)}')
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05) #model mare, 0.8
        optimizer.step()

        total_loss += loss.item()

    time_end = time.perf_counter()

    avg_loss = total_loss / num_batches
    print(f"Train loss for epoch {epoch}: {avg_loss}, duration {time_end-time_begin} seconds")
    train_loss.append(avg_loss)
    test_loss = test_model(test_loader, model, loss_function)

    if test_loss < minimum_testing_error:
        minimum_testing_error = test_loss
        minimum_model = copy.deepcopy(model.state_dict())
        minimum_epoch = epoch

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"intermediary_results\\saved_checkpoint_{model.__class__.__name__}_{epoch}_epoch.pth")
        torch.save(minimum_model, f"best_model_{model.__class__.__name__}.pth")
        

    validation_loss.append(test_loss)
    
torch.save(model.state_dict(), "saved.pth")
print(f"Minimum epoch {minimum_epoch}, minimum loss {minimum_testing_error}")
torch.save(minimum_model, f"best_model_{model.__class__.__name__}.pth")


plt.plot(epochs, train_loss)
plt.plot(epochs, validation_loss)
plt.show()
