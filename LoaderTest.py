from utils_detection import *
import os
from torch.utils.data import Dataset
import torch
import random

class DatasetTest(Dataset):
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
    
        self.X = torch.tensor(self.X).float().to(device)
        self.Y = torch.tensor(self.Y).long().to(device)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y