from torch import nn
import torch

class LSTM_for_best(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_for_best, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 10

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.3)
        
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
    
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 5

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.4)
        
        self.fc_1  = nn.Linear(self.hidden_units, 128)
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
        out = self.fc_final(out)

        return out
    
    def return_train_data():
        SEQUENCE_LENGTH = 16
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE
    
class LSTM_deep_sequence(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_deep_sequence, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 10

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.4)
        
        self.fc_1  = nn.Linear(self.hidden_units, 128)
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
        out = self.fc_final(out)

        return out
    
    def return_train_data():
        SEQUENCE_LENGTH = 16
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE
    
class LSTM_shallow(nn.Module):
    #best results!
    #batch size 64
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_shallow, self).__init__()

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
    
class LSTM_very_deep(nn.Module):
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_very_deep, self).__init__()

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
        HIDDEN_SIZE = 1024

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE
    
class LSTM_best_but_wider(nn.Module):
    #acuratete 0.46
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_best_but_wider, self).__init__()

        self.num_features = num_features
        self.num_classes = 12
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.num_layers = 2

        self.lstm = nn.LSTM(input_size = self.num_features, 
                            hidden_size = self.hidden_units, num_layers = self.num_layers, 
                            batch_first = True,
                            dropout = 0.6)
        
        self.fc_1  = nn.Linear(self.hidden_units, 512)
        self.fc_2  = nn.Linear(self.hidden_units, 256)
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
        out  = self.fc_2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_final(out)

        return out
    
    def return_train_data():
        SEQUENCE_LENGTH = 16
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE

class LSTM_shallow_20_sequence(nn.Module):
    #batch size 64
    def __init__(self, num_features, hidden_units, seq_length):
        super(LSTM_shallow_20_sequence, self).__init__()

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
        SEQUENCE_LENGTH = 20
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE
    
class LSTM(nn.Module):
    #batch size 128
    #optimizer = torch.optim.Adam(model.parameters(), lr = e1-4)
    #cliiping 0.05
    #best!
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
                            dropout = 0.7)
        
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
        SEQUENCE_LENGTH = 23
        INPUT_SIZE = 34
        HIDDEN_SIZE = 512

        return SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE
