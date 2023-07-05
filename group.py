import umap
import torch
import umap.plot
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.utils.data as data

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
from losses import SupConLoss, ContrastiveLoss
from torch.nn.functional import normalize
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, MeanShift, DBSCAN
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
# import scanpy as sc

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden),
                                    nn.BatchNorm1d(hidden),
                                    nn.ReLU(inplace=True),
                                    # nn.Linear(hidden,emb*2),
                                    # nn.BatchNorm1d(emb*2),
                                    # nn.ReLU(inplace=True),
                                    nn.Linear(hidden,emb),
                                    nn.BatchNorm1d(emb),
                                    nn.ReLU(inplace=True),

          )
    def forward(self,x):
        return self.encoder(x)
    
class SimCLR(nn.Module):
    def __init__(self, hidden, emb, input_dim):
        super().__init__()
        self.backbone = Encoder(input_dim, hidden=hidden, emb=emb)
        self.projection = nn.Sequential(
            nn.Linear(in_features=emb, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        return  F.normalize(self.projection(embedding),p=2, dim=1)
    
class Gene_data(Dataset):
 
    def __init__(self,X_train, y_train, transform=None):
        if torch.is_tensor(X_train):
            self.x_train = normalize(torch.log(X_train+1), p=2.0, dim = 0)
            self.y_train=torch.tensor(y_train,dtype=torch.int32)
        else:
            x = torch.tensor(X_train.values,dtype=torch.float32)
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            self.y_train=torch.tensor(y_train.values,dtype=torch.int32)
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        x = self.x_train[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y_train[idx]

@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model 
    network = deepcopy(model.backbone)
    network.train()
    network.to('cuda')

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to('cuda')
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return feats, labels


def clustering_dbscan(X, eps, min_samples):
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(X)
    print('Found clusters', len(np.unique(dbs.labels_)))
    return dbs.labels_

if __name__ == "__main__":
    cudnn.deterministic = True
    cudnn.benchmark = True
    NUM_WORKERS = 1
    emb = 128
    hidden = 1024
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 1000
    device='cuda'
    X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')
    input_dim = X.shape[1]
    genes = X.columns
    allen_x, allen_y = torch.load('preprocessed/ax.pt'), torch.load('preprocessed/ay.pt')
    gide_x, gide_y =  torch.load('preprocessed/gx.pt'), torch.load('preprocessed/gy.pt')
    liu_x, liu_y =  torch.load('preprocessed/lx.pt'), torch.load('preprocessed/ly.pt')
    riaz_x, riaz_y=  torch.load('preprocessed/rx.pt'), torch.load('preprocessed/ry.pt')
    hugo_x, hugo_y=  torch.load('preprocessed/hx.pt'), torch.load('preprocessed/hy.pt')
    lee_x, lee_y=  torch.load('preprocessed/leex.pt'), torch.load('preprocessed/leey.pt')
    allen_x = allen_x[genes]
    gide_x = gide_x[genes]
    liu_x = liu_x[genes]
    riaz_x = riaz_x[genes]
    hugo_x = hugo_x[genes]
    lee_x = lee_x[genes]
    features_all = {}
    labels_all = {}
    allen_data = Gene_data(allen_x, allen_y)
    riaz_data = Gene_data(riaz_x, riaz_y)
    gide_data = Gene_data(gide_x, gide_y)
    liu_data = Gene_data(liu_x, liu_y)
    hugo_data = Gene_data(hugo_x, hugo_y)
    lee_data = Gene_data(lee_x, lee_y)
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    model.projection = nn.Identity()
    model.load_state_dict(torch.load('sim.pth'))
    fg, lg = prepare_data_features(model, gide_data)
    fa, la = prepare_data_features(model, allen_data)
    fl, ll = prepare_data_features(model, liu_data)
    fr, lriaz = prepare_data_features(model, riaz_data)
    fh, lh = prepare_data_features(model, hugo_data)
    flee, llee = prepare_data_features(model, lee_data)
    #### 
    # fg, lg = gide_data.x_train, gide_data.y_train
    # fa, la = allen_data.x_train, allen_data.y_train
    # fl, ll = liu_data.x_train, liu_data.y_train
    # fr, lriaz = riaz_data.x_train, riaz_data.y_train
    # fh, lh = hugo_data.x_train, hugo_data.y_train
    # flee, llee = lee_data.x_train, lee_data.y_train
    features_all['r'] = fr
    labels_all['r'] = lriaz
    features_all['a'] = fa
    labels_all['a'] = la
    features_all['g'] = fg
    labels_all['g'] = lg
    features_all['l'] = fl
    labels_all['l'] = ll 
    features_all['h'] = fh
    labels_all['h'] = lh
    features_all['lee'] = flee
    labels_all['lee'] = llee
    fea = torch.vstack([fa, fg, fr, fh, fl,flee])
    l1 = [str(la[i].item()) for i in range(len(fa))] + [str(lg[i].item()) for i in range(len(fg))]\
         + [str(lriaz[i].item()) for i in range(len(fr))] \
        +[str(lh[i].item()) for i in range(len(fh))] +[str(ll[i].item()) for i in range(len(fl))] \
        + [str(llee[i].item()) for i in range(len(flee))]

    l2 = ['allen:' for i in range(len(fa))] + ['gide:' for i in range(len(fg))]\
         + ['riaz:' for i in range(len(fr))]\
        +['hugo:' for i in range(len(fh))] + ['liu:' for i in range(len(fl))]\
        + ['lee:' for i in range(len(flee))]

    mapper = umap.UMAP(n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,).fit(fea.to('cpu'))
    umap.plot.points(mapper,labels=np.array(l1),color_key_cmap='Paired', theme='fire')
    umap.plot.points(mapper,labels=np.array(l2),color_key_cmap='Paired', theme='fire')

    clusterable_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
        ).fit_transform(fea.to('cpu'))

    labelsDBSCAN = clustering_dbscan(mapper.embedding_, eps=0.5, min_samples=5)
    # la = np.array(labelsDBSCAN)
    # lm1 = np.where(la==-1)
    # l0 = np.where(la==0)
    # l1 = np.where(la==1)
    # l2 = np.where(la==2)
    # l3 = np.where(la==3)
    # l4 = np.where(la==4)

    # la = list(labelsDBSCAN)
    # la[l4[:]] = 'res'
    newL = []
    for i in (labelsDBSCAN):
        if i==-1:
            newL.append('mix')
        if i==0:
            newL.append('mix')
        if i==1:
            newL.append('NR')
        if i==2:
            newL.append('NR')
        if i==3:
            newL.append('mix')
        if i==4:
            newL.append('R')

    fig, ax = plt.subplots()    
    scatter = ax.scatter(
        clusterable_embedding[:,0], clusterable_embedding[:,1],c=labelsDBSCAN,
        cmap='tab20', alpha=0.7, s = 10
    )
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Cluster")
    ax.add_artist(legend1)
    plt.show()



    pid = list(allen_x.index.values)+list(gide_x.index.values)+list(riaz_x.index.values)\
    +list(hugo_x.index.values)+list(liu_x.index.values)+list(lee_x.index.values)
    pdf = pd.concat([allen_x,gide_x, riaz_x, hugo_x, liu_x, lee_x])
    ldf =  pd.concat([allen_y,gide_y, riaz_y, hugo_y, liu_y, lee_y])



