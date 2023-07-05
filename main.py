#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:57:32 2023

@author: huan
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('preprocessed/allen.csv',index_col=0)
df1 = df1.set_index('index')
df1.head()


df2 = pd.read_csv('preprocessed/gide.csv',index_col=0)
df2 = df2.set_index('index')
df2.head()
df2.loc[df2["response"] == "PR",'response'] = 1
df2.loc[df2["response"] == "CR",'response'] = 1
df2.loc[df2["response"] == "PD",'response'] = 0
df2.loc[df2["response"] == "SD",'response'] = 0

df3 = pd.read_csv('preprocessed/liu.csv',index_col=0)
df3 = df3.set_index('index')
df3.head()
df3.loc[df3["response"] == "PR",'response'] = 1
df3.loc[df3["response"] == "CR",'response'] = 1
df3.loc[df3["response"] == "PD",'response'] = 0
df3.loc[df3["response"] == "SD",'response'] = 0
df3.loc[df3["response"] == "MR",'response'] = 0

df4 = pd.read_csv('preprocessed/riaz.csv',index_col=0)
df4 = df4.set_index('index')
df4.head()
df4.loc[df4["response"] == "PRCR",'response'] = 1
df4.loc[df4["response"] == "PD",'response'] = 0
df4.loc[df4["response"] == "SD",'response'] = 0

df5 = pd.read_csv('preprocessed/Hugo_TPM.csv',index_col=0)
df5 = df5.T
cli = pd.read_csv('preprocessed/Hugo_clinical.csv')
df5['response'] = 0
for ii in df5.index:
    row = df5.loc[ii]
    res = cli[cli['Patient.ID']==ii]['Response']
    if res.values=='NR':
        res = 0
    else:
        res = 1
    df5.loc[ii,'response'] = res

df6 = pd.read_csv('preprocessed/Lee_TPM.csv', index_col=0)
df6 = df6.T
cli_lee = pd.read_csv('preprocessed/Lee_clinical.csv', index_col=0)
aa = list(cli_lee.Pt_ID.values)
df6 = df6[df6.index.isin(aa)]
df6['response'] = 0
for p in df6.index:
    res = cli_lee[cli_lee.Pt_ID==p]['Resp_NoResp'].values
    if res =='Response':
        res = 1
    elif res =='No_Response':
        res = 0
    else:
        print("!")
    df6.loc[p,'response'] = res


g1 = df1.columns
g2 = df2.columns
g3 = df3.columns
g4 = df4.columns
g5 = df5.columns
g6 = df6.columns
gg = g1.intersection(g2)
gg = gg.intersection(g3)
gg = gg.intersection(g4)
gg = gg.intersection(g5)
gg = gg.intersection(g6)
df1 = df1[gg]
df2 = df2[gg]
df4 = df4[gg]
df3 = df3[gg]
df5 = df5[gg]
df6 = df6[gg]

df4 = df4.dropna(axis=0)

frames = [df1, df2, df3, df4, df5]

result = pd.concat(frames)
result.head()


def drop_almost_zero(df, percentage):
    column_cut_off = int(percentage/100*len(df)) 
    b = (df == 0).sum(axis='rows')
    df = df[ b[ b <= column_cut_off].index.values ]
    return df


X = result[result.columns[0:-1]]
row_count = X.shape[0]
columns_to_drop = []
for column, count in X.apply(lambda column: (column == 0).sum()).iteritems():
    if count / row_count >= 0.6:
        columns_to_drop.append(column)
X.drop(columns_to_drop, axis=1, inplace=True)
len_feature = X.shape[1]

y = result[result.columns[-1]].astype('int32')
torch.save(X,'preprocessed/X.pt')
torch.save(y,'preprocessed/y.pt')

df1.drop(columns_to_drop, axis=1, inplace=True)
X = df1[df1.columns[0:-1]]
y = df1[df1.columns[-1]].astype('int32')
torch.save(X,'preprocessed/ax.pt')
torch.save(y,'preprocessed/ay.pt')

df2.drop(columns_to_drop, axis=1, inplace=True)
X = df2[df2.columns[0:-1]]
y = df2[df2.columns[-1]].astype('int32')
torch.save(X,'preprocessed/gx.pt')
torch.save(y,'preprocessed/gy.pt')

df3.drop(columns_to_drop, axis=1, inplace=True)
X = df3[df3.columns[0:-1]]
y = df3[df3.columns[-1]].astype('int32')
torch.save(X,'preprocessed/lx.pt')
torch.save(y,'preprocessed/ly.pt')

df4.drop(columns_to_drop, axis=1, inplace=True)
X = df4[df4.columns[0:-1]]
y = df4[df4.columns[-1]].astype('int32')
torch.save(X,'preprocessed/rx.pt')
torch.save(y,'preprocessed/ry.pt')

df5.drop(columns_to_drop, axis=1, inplace=True)
X = df5[df5.columns[0:-1]]
y = df5[df5.columns[-1]].astype('int32')
torch.save(X,'preprocessed/hx.pt')
torch.save(y,'preprocessed/hy.pt')


df6.drop(columns_to_drop, axis=1, inplace=True)
X = df6[df6.columns[0:-1]]
y = df6[df6.columns[-1]].astype('int32')
torch.save(X,'preprocessed/leex.pt')
torch.save(y,'preprocessed/leey.pt')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# gene_train = Gene_data(X_train, y_train)
# train_loader=DataLoader(gene_train, batch_size=32, shuffle=True)

# gene_test = Gene_data(X_test, y_test)
# test_loader=DataLoader(gene_test, batch_size=32, shuffle=True)


# class Net(nn.Module):
#   def __init__(self,input_shape):
#     super(Net,self).__init__()
#     self.fc1 = nn.Linear(input_shape,256)
#     self.fc2 = nn.Linear(256,128)
#     self.fc3 = nn.Linear(128,2)
#   def forward(self,x):
#     x = torch.relu(self.fc1(x))
#     x = torch.relu(self.fc2(x))
#     x = F.log_softmax(self.fc3(x),dim = 1)
#     return x

# device='cuda'
# model = Net(input_shape=len_feature).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# def accuracy(out, labels):
#     _,pred = torch.max(out, dim=1)
#     return torch.sum(pred==labels).item()

# n_epochs = 50
# print_every = 10
# valid_loss_min = np.Inf
# val_loss = []
# val_acc = []
# train_loss = []
# train_acc = []
# total_step = len(train_loader)
# for epoch in range(1, n_epochs+1):
#     running_loss = 0.0
#     # scheduler.step(epoch)
#     correct = 0
#     total=0
#     print(f'Epoch {epoch}\n')
#     for batch_idx, (data_, target_) in enumerate(train_loader):
#         target_ = target_.type(torch.LongTensor)
#         data_, target_ = data_.to(device), target_.to(device)# on GPU
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = model(data_)
#         loss = criterion(outputs, target_)
#         loss.backward()
#         optimizer.step()
#         # print statistics
#         running_loss += loss.item()
#         _,pred = torch.max(outputs, dim=1)
#         correct += torch.sum(pred==target_).item()
#         total += target_.size(0)
#         if (batch_idx) % 20 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
#     train_acc.append(100 * correct / total)
#     train_loss.append(running_loss/total_step)
#     print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
#     batch_loss = 0
#     total_t=0
#     correct_t=0
#     with torch.no_grad():
#         model.eval()
#         for data_t, target_t in (test_loader):
#             target_t = target_t.type(torch.LongTensor)
#             data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
#             outputs_t = model(data_t)
#             loss_t = criterion(outputs_t, target_t)
#             batch_loss += loss_t.item()
#             _,pred_t = torch.max(outputs_t, dim=1)
#             correct_t += torch.sum(pred_t==target_t).item()
#             total_t += target_t.size(0)
#         val_acc.append(100 * correct_t / total_t)
#         val_loss.append(batch_loss/len(test_loader))
#         network_learned = batch_loss < valid_loss_min
#         print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
