from utils import load_data, gene_signatures
from aug import domain_aug, Gaussian,RandomMask,FixedSigAug, RandomSigAug
import pandas as pd
import numpy as np
import umap
import torch
import umap.plot

X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')
tcga = torch.load('tcga_skcm.pt')
duplicate_cols = tcga.columns[tcga.columns.duplicated()]
tcga.drop(columns=duplicate_cols, inplace=True)
t5 = np.loadtxt('top10k.txt')
X = X.iloc[:, t5]
cc =  X.columns.intersection(tcga.columns)
X = X[cc] 
genes = X.columns
tcga = tcga[genes]
signs = gene_signatures(genes).squeeze()




i=1
features_all = {}
labels_all = {}
allen_data, riaz_data, gide_data,  liu_data, hugo_data ,lee_data = load_data(genes, None)
fa, la = allen_data.x_train, allen_data.y_train
fg, lg = gide_data.x_train, gide_data.y_train
fl, ll = liu_data.x_train, liu_data.y_train
fh, lh = hugo_data.x_train, hugo_data.y_train
flee, llee = lee_data.x_train, lee_data.y_train
fr, lriaz = riaz_data.x_train, riaz_data.y_train
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
all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]

train = all_data[:i] + all_data[i+1:]
d = [train[i].x_train for i in range(5)]
y = [train[i].y_train for i in range(5)]
r = [np.where(y[i]==1)[0] for i in range(5)]
nr = [np.where(y[i]==0)[0] for i in range(5)]
dim = len(y[i])
newd = []
newy = []
for ns in range(200):
    idx = np.random.choice(dim)
    x = d[i][idx]
    yi = y[i][idx]
    # lam1 = torch.rand(5)
    # lam1 /= torch.sum(lam1) 
    lam1 = torch.tensor(np.random.dirichlet(np.ones(5),size=1))[0]
    if yi ==1:
        m1 = [d[i][np.random.choice(r[i])] * lam1[i] for i in range(5)]
    else:
        m1 = [d[i][np.random.choice(nr[i])] * lam1[i] for i in range(5)]
    mix = sum(m1)
    xa = 0.3 * x + 0.7 * mix
    newd.append(xa)
    newy.append(yi)
xd = torch.vstack(newd)
yd = torch.vstack(newy)
features_all['dm'] = xd
labels_all['dm'] = yd
fea = torch.vstack([fa, fg, fr, fh, fl,flee, xd])


l11 = [str(la[i].item()) for i in range(len(fa))] + [str(lg[i].item()) for i in range(len(fg))]\
        + [str(lriaz[i].item()) for i in range(len(fr))] \
    +[str(lh[i].item()) for i in range(len(fh))] +[str(ll[i].item()) for i in range(len(fl))] \
    + [str(llee[i].item()) for i in range(len(flee))]+[str(yd[i].item()) for i in range(len(xd))] 

l22 = ['allen:' for i in range(len(fa))] + ['gide:' for i in range(len(fg))]\
        + ['riaz:' for i in range(len(fr))]\
    +['hugo:' for i in range(len(fh))] + ['liu:' for i in range(len(fl))]\
    + ['lee:' for i in range(len(flee))] + ['Domain:' for i in range(len(yd))] 


mapper = umap.UMAP(n_neighbors=15,
    min_dist=0.0,
    n_components=2,
    metric='euclidean',
    random_state=42).fit(fea.to('cpu'))
fea = torch.vstack([fa, fg, fr, fh, fl,flee])
l11 = [str(la[i].item()) for i in range(len(fa))] + [str(lg[i].item()) for i in range(len(fg))]\
        + [str(lriaz[i].item()) for i in range(len(fr))] \
    +[str(lh[i].item()) for i in range(len(fh))] +[str(ll[i].item()) for i in range(len(fl))] \
    + [str(llee[i].item()) for i in range(len(flee))]

l22 = ['allen:' for i in range(len(fa))] + ['gide:' for i in range(len(fg))]\
        + ['riaz:' for i in range(len(fr))]\
    +['hugo:' for i in range(len(fh))] + ['liu:' for i in range(len(fl))]\
    + ['lee:' for i in range(len(flee))] 
mapper = umap.UMAP(n_neighbors=15,
    min_dist=0.0,
    n_components=2,
    metric='euclidean',
    random_state=42).fit(fea.to('cpu'))

umap.plot.points(mapper,labels=np.array(l11),theme='fire')
umap.plot.points(mapper,labels=np.array(l22),theme='fire')




