import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return self.data.shape[0]
    
class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = nn.Linear(self.d_model, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        if self.verbose:
            print('input (n_samples, n_channel, n_length)', out.shape)
        out = out.permute(2, 0, 1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out.shape)

        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder', out.shape)

        out = out.mean(0)
        if self.verbose:
            print('global pooling', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)

        return out    

n_samples = 32
n_length = 1000
n_channel = 1
n_classes = 2 
input_sequence = torch.rand(32, 1, 1000).to('cuda')  # batch_size x seq_length
model = Transformer1d(
    n_classes=n_classes, 
    n_length=n_length, 
    d_model=n_channel, 
    nhead=1, 
    dim_feedforward=256, 
    dropout=0.1, 
    activation='relu',verbose=True)
model.to('cuda')
output = model(input_sequence)
print(output.shape)