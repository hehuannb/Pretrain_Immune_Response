
import os
import umap
import torch
import umap.plot
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from aug import Gaussian, RandomMask, ImmuneAug,tabaug,same
import pytorch_lightning as pl
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay
from scipy.stats import norm
from losses import SupConLoss, ContrastiveLoss
from torch.nn.functional import normalize
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from model import ResNet1D
from sklearn.cluster import KMeans
from pytorch_metric_learning import losses,miners


def eval(v, tmp, lr, epoch, mode='lp'):
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    model.load_state_dict(torch.load('simclr.pt'))

    classifier = Net(emb, tmp).to(device)
    criterion = nn.CrossEntropyLoss()
    if mode=='lp':
        model.projection = nn.Identity()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        model.eval()
    elif mode=='ft':
        model.projection = classifier
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise('not implemented')
    train_loss = []
    train_acc = []
    for _ in range(epoch):
        batch, labels = v.dataset.x_train, v.dataset.y_train
        labels = labels.type(torch.LongTensor)
        batch, labels = batch.to(device), labels.to(device)# on GPU
        optimizer.zero_grad()
        if mode == 'lp':
            with torch.no_grad():
                feats = model(batch)
            outputs = classifier(feats)
        elif mode=='ft':
            outputs = model(batch)
        else:
            raise('not implemented')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _,pred = torch.max(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), pred.cpu())
        train_acc.append(acc)
        train_loss.append(loss.item())
        # print(acc)
    return model, classifier

def report(tmp, lr, epoch, label_cohorts, mode='lp'):
    all_cohorts = {'a':allen_loader,'g':gide_loader,\
                   'l':liu_loader,'r':riaz_loader,'h':hugo_loader}
    for k, v in label_cohorts.items():
        model, classifier = eval(v, tmp, lr, epoch, mode)
        classifier.eval()
        model.eval()
        for k2, v2 in all_cohorts.items():
            data_, target_ = v2.dataset.x_train, v2.dataset.y_train
            data_, target_ = data_.to(device), target_.to(device)# on GPU
            if mode == 'lp':
                feats = model(data_)
                outputs = classifier(feats)
            elif mode=='ft':
                outputs = model(data_)
            else:
                raise('not implemented')
            _,pred = torch.max(outputs, dim=1)
            target_ = target_.cpu()
            pred = pred.cpu()
            f1 = f1_score(target_, pred)
            roc_auc = roc_auc_score(target_, pred)
            # cm = confusion_matrix(target_.cpu(), pred.cpu())
            # ConfusionMatrixDisplay(cm).plot()
            print(f"From {k} to {k2}: Acc={accuracy_score(target_, pred):.3f}, F1={f1:.3f}, auc={roc_auc:.3f}")
            # RocCurveDisplay.from_predictions( target_.cpu(), pred.cpu())
        print("----------")

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def define_param_groups(model, weight_decay, optimizer_name):
   def exclude_from_wd_and_adaptation(name):
       if 'bn' in name:
           return True
       if optimizer_name == 'lars' and 'bias' in name:
           return True

   param_groups = [
       {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
       },
       {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
       },
   ]
   return param_groups


def vis():
    features_all = {}
    labels_all = {}
    allen_data = Gene_data(allen_x, allen_y)
    riaz_data = Gene_data(riaz_x, riaz_y)
    gide_data = Gene_data(gide_x, gide_y)
    liu_data = Gene_data(liu_x, liu_y)
    hugo_data = Gene_data(hugo_x, hugo_y)
    fg, lg = prepare_data_features(model, gide_data)
    fa, la = prepare_data_features(model, allen_data)
    fl, ll = prepare_data_features(model, liu_data)
    fr, lriaz = prepare_data_features(model, riaz_data)
    fh, lh = prepare_data_features(model, hugo_data)
     
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
    fea = torch.vstack([fa, fg, fr, fh])
    l1 = [str(la[i].item()) for i in range(len(fa))] + [str(lg[i].item()) for i in range(len(fg))]\
         + [str(lriaz[i].item()) for i in range(len(fr))] \
        +[str(lh[i].item()) for i in range(len(fh))]

    l2 = ['allen:' for i in range(len(fa))] + ['gide:' for i in range(len(fg))]\
         + ['riaz:' for i in range(len(fr))]\
        +['hugo:' for i in range(len(fh))]

    mapper = umap.UMAP().fit(fea.to('cpu'))
    umap.plot.points(mapper,labels=np.array(l1),theme='fire')

    umap.plot.points(mapper,labels=np.array(l2),theme='fire')


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden),
                                    # nn.BatchNorm1d(hidden),
                                    nn.LayerNorm(hidden, 1e-3),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden,emb),
                                    # nn.BatchNorm1d(emb),
                                    nn.LayerNorm(emb, 1e-3),
                                    nn.ReLU(inplace=True),
          )
    def forward(self,x):
        return self.encoder(x)
    
class SimCLR(nn.Module):
    def __init__(self, hidden, emb, input_dim=1):
        super().__init__()
        kernel_size = 16
        stride = 2
        n_block = 8
        downsample_gap = 2
        increasefilter_gap = 12  
        embedding = emb  
        # self.backbone = ResNet1D(
        #     in_channels=1, 
        #     base_filters= 64, # 64 for ResNet1D, 352 for ResNeXt1D
        #     kernel_size=kernel_size, 
        #     stride=stride, 
        #     groups=4, 
        #     n_block=n_block, 
        #     n_classes=embedding, 
        #     downsample_gap=downsample_gap, 
        #     increasefilter_gap=increasefilter_gap, 
        #     use_do=False)
        self.backbone = Encoder(input_dim, hidden=hidden, emb=emb)
        self.projection = nn.Sequential(
            nn.Linear(in_features=emb, out_features=128),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(128),
            # nn.LayerNorm(128),
            # nn.Tanh(),
            # nn.Linear(in_features=128, out_features=128),
            # nn.BatchNorm1d(emb),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        return  F.normalize(self.projection(embedding),p=2, dim=1)
    
class Net(nn.Module):
    def __init__(self,input_shape, temp):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,2)
        self.n1 = nn.Linear(input_shape,2)
        self.tmp = temp
        
    def forward(self,x):
        # x = torch.tanh(self.bn1(self.fc1(x)))
        # x = torch.tanh(self.fc2(x))
        # return x
        return self.n1(x)/self.tmp


class Gene_data(Dataset):
 
    def __init__(self,X_train, y_train, transform=None):
        x = torch.tensor(X_train.values,dtype=torch.float32)
        # self.x_train= torch.log(normalize(torch.tensor(X_train.values,dtype=torch.float32)+1,p=2.0, dim = 0)) 
        self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
        # self.x_train = torch.log(x+1)
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
    network.projection = nn.Identity()  # Removing projection head g(.)
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




@torch.no_grad()
def pseduo_label(features_un, f2, contrast_feature, labels1, topk):
    fp, fn = contrast_feature[torch.where(labels1==1)], contrast_feature[torch.where(labels1==0)]
    fpc, fnc = fp.mean(0), fn.mean(0)
    distP = F.cosine_similarity(features_un,fpc)
    distP2 = F.cosine_similarity(f2,fpc)
    dpm = (distP+distP2)/2
    distN = F.cosine_similarity(features_un,fnc)
    distN2 = F.cosine_similarity(f2,fnc)
    dnm = (distN+distN2)/2
    index_sortedP = torch.argsort(dpm)
    index_sortedN = torch.argsort(dnm)
    top_idx_p = index_sortedP[:topk]
    top_idx_n = index_sortedN[:topk]
    pl = torch.cat([torch.ones(len(top_idx_p)), torch.zeros(len(top_idx_n))])
    fea_pl = torch.cat([features_un[top_idx_p].unsqueeze(1), \
                        f2[top_idx_p].unsqueeze(1)], dim=1)
    fea_nl = torch.cat([features_un[top_idx_n].unsqueeze(1), \
                        f2[top_idx_n].unsqueeze(1)], dim=1)

    return torch.cat([fea_pl,fea_nl]), pl.to('cuda')

if __name__ == "__main__":
    cudnn.deterministic = True
    cudnn.benchmark = True
    NUM_WORKERS = 1
    emb = 100
    hidden = 5000
    lr = 5e-4
    weight_decay = 1e-4
    max_epochs = 200
    device='cuda'
    X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')
    immune_gene = pd.read_csv('/home/huan/Downloads/InnateDB_genes.csv')
    immunegene = X.columns.intersection(immune_gene.name)
    # X = X[immunegene]
    genes = X.columns
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    loc = [X.columns.get_loc(c) for c in immunegene if c in X]
    mask= np.full(len(X.columns),False,dtype=bool)
    mask[loc]=True
    input_dim = X.shape[1]
    xx = np.log(X.values+1)
    xx = (xx - xx.mean(0))/(xx.std(0))
    mean_list = torch.tensor([norm.fit(xx[:,i])[0] for i in range(input_dim)]).to(torch.float32)
    std_list = torch.tensor([norm.fit(xx[:,i])[1] for i in range(input_dim)]).to(torch.float32)
    # contrast_transforms = transforms.Compose([Gaussian(), RandomMask(), tabaug(mean_list,std_list)])
    # contrast_transforms = transforms.Compose([transforms.RandomApply([Gaussian()],p=0.5), \
    #                                          transforms.RandomApply([RandomMask()],p=0.5)])
    
    contrast_transforms = transforms.RandomChoice([Gaussian(std=0.1), RandomMask(p=0.2)])
    train_data = Gene_data(X, y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    allen_x, allen_y = torch.load('preprocessed/ax.pt'), torch.load('preprocessed/ay.pt')
    gide_x, gide_y =  torch.load('preprocessed/gx.pt'), torch.load('preprocessed/gy.pt')
    liu_x, liu_y =  torch.load('preprocessed/lx.pt'), torch.load('preprocessed/ly.pt')
    riaz_x, riaz_y=  torch.load('preprocessed/rx.pt'), torch.load('preprocessed/ry.pt')
    hugo_x, hugo_y=  torch.load('preprocessed/hx.pt'), torch.load('preprocessed/hy.pt')
    allen_x = allen_x[genes]
    gide_x = gide_x[genes]
    liu_x = liu_x[genes]
    riaz_x = riaz_x[genes]
    hugo_x = hugo_x[genes]

    allen_data = Gene_data(allen_x, allen_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    riaz_data = Gene_data(riaz_x, riaz_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    gide_data = Gene_data(gide_x, gide_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    liu_data = Gene_data(liu_x, liu_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    hugo_data = Gene_data(hugo_x, hugo_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))

    # test_data = Gene_data(X_test, y_test, transform=ContrastiveLearningViewGenerator(contrast_transforms))

    batch_size = 64
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    liu_loader = data.DataLoader(liu_data, batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    gide_loader = data.DataLoader(gide_data, batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    riaz_loader = data.DataLoader(riaz_data, batch_size=32, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    allen_loader = data.DataLoader(allen_data, batch_size=32, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    hugo_loader = data.DataLoader(hugo_data, batch_size=13, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    param_groups = define_param_groups(model, weight_decay, 'adam')
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(range(max_epochs)), eta_min=0,
    #                                                        last_epoch=-1)
    tmp = 0.2
    criterion = SupConLoss(temperature= tmp)

    c2 = ContrastiveLoss(batch_size, tmp)
    labelX = pd.concat([liu_x])
    labely = pd.concat([liu_y])
    llll_data = Gene_data(labelX, labely, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    label_loader = data.DataLoader(llll_data, batch_size=batch_size, shuffle=True,
                                drop_last=False, pin_memory=True, num_workers=NUM_WORKERS) 
    all_iter  = iter(train_loader)
    for epoch_counter in tqdm(range(max_epochs)):
        for _, (batch1,labels1) in enumerate(label_loader):
            try:
                batch_all, _  = next(all_iter)
            except StopIteration:
                all_iter  = iter(train_loader)
                batch_all, _  = next(all_iter)
            batch = torch.cat(batch1, dim=0).to('cuda')
            labels1 = labels1.type(torch.LongTensor).to('cuda')
            feats = model(batch)
            f1,f2 = torch.chunk(feats,2, dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            batch_a = torch.cat(batch_all, dim=0).to('cuda')
            feats = model(batch_a)
            f1_u,f2_u = torch.chunk(feats,2, dim=0)
            feat_pl, pl = pseduo_label(f1_u, f2_u, contrast_feature, labels1,10)
            # ff1, ff2 = torch.unbind(feat_pl, dim=1)
            # fp = torch.cat([f1, ff1], dim=0)
            # fn = torch.cat([f2, ff2], dim=0)
            # embedding = torch.cat([fp.unsqueeze(1),fn.unsqueeze(1)],dim=1)
            # glabel = torch.cat([labels1, pl])
            embedding = torch.cat([features,feat_pl],dim=0)
            glabel = torch.cat([labels1, pl])
            # embeddings = torch.cat(torch.unbind(features, dim=1), dim=0)
            # hard_pairs = miner(features, labels1)
            # loss = criterion(features, labels1, hard_pairs)
            loss = criterion(embedding, glabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        if epoch_counter % 25==0:
            print(loss.item())
    torch.save(model.state_dict(), 'simclr.pt')

    l_cohorts = {'label':label_loader}
    report(0.01, 1e-3, 300,l_cohorts,'ft')
    print('llllll')
    # report(0.1, 1e-3, 200,l_cohorts,'ft')
    vis()
