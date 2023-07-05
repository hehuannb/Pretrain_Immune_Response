import pandas as pd
import torch
import numpy as np
import os

# data = pd.DataFrame(columns=['gene_name','tpm_unstranded'])
i = 0
X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')


for root, subdirs,f in os.walk('data/luda/luad'):
    if f[0][-3:] == 'tsv':
        file_path = os.path.join(root, f[0])
        df = pd.read_csv(file_path,sep='\t',header=1)
        df = df[['gene_name','tpm_unstranded']]
        df = df.iloc[4:,]
        # df = df.reset_index(drop=True)
        df = df.set_index(df.columns[0])
        # df = df.reset_index(drop=True)
        df = df.T
        if i ==0:
            data = df
        else:
            data = pd.concat([data, df])
        print(data.shape)
        i +=1 
data.index = np.arange(data.shape[0])
gg = data.columns.intersection(X.columns)
data = data[gg]
torch.save(data,'tcga_luda.pt')