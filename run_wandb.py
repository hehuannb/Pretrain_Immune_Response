import wandb
import os
import umap
import torch
import umap.plot
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import torch.utils.data as data
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay
from torch.nn.functional import normalize
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
from pytorch_metric_learning import losses
from scipy import stats
import matplotlib.pyplot as plt
from utils import load_data, gene_signatures
from trainer import SimCLR, Trainer, classifier
from aug import domain_aug, Gaussian,RandomMask,FixedSigAug, RandomSigAug

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x, y, x2):
        return [self.base_transform(x, y, x2) for i in range(self.n_views)]
    
class Gene_data(Dataset):
    def __init__(self, X_train, y_train, transform=None):
        if torch.is_tensor(X_train):
            x = X_train.clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            self.y_train = torch.tensor(y_train,dtype=torch.int32)
        else:
            x = torch.tensor(X_train.values,dtype=torch.float32).clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            self.y_train = torch.tensor(y_train.values,dtype=torch.int32)
        self.transform = transform
        self.ridx = np.where(self.y_train==1)[0]
        self.nridx = np.where(self.y_train==0)[0]
        self.test = 0
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        if self.transform is not None:
            if y ==1:
                mixup_idx = np.random.choice(self.ridx)
            else:
                mixup_idx = np.random.choice(self.nridx)
            x2 = self.x_train[mixup_idx]
            x = self.transform(x, y, x2)
            # x = [self.transform[0](x,y, x2), self.transform[1](x, y, x2)]
        return x, y
    
if __name__ == "__main__":
    # Key hyper parameters
    cudnn.deterministic = True
    cudnn.benchmark = True
    NUM_WORKERS = 1
    feat_dim = 128
    proj_dim = 32
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 500
    device='cuda'
    tmp = 0.1
    batch_size = 64 

    # Load Datasets
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
    allen_data, riaz_data, gide_data,  liu_data, hugo_data ,lee_data = load_data(genes, None)
    all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]
    names = ['allen','riaz','gide','liu','hugo','lee']
    train_loaders = []
    val_loaders = []
    
    for i in range(6):
        train = all_data[:i] + all_data[i+1:]
        X = torch.cat([x.x_train for x in train])
        y = torch.cat([x.y_train for x in train])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        input_dim = X_train.shape[1]
        # contrast_transforms = [domain_aug(genes, i), RandomSigAug(signs, 0.5)]
        contrast_transforms = transforms.RandomChoice([domain_aug(genes, i), RandomSigAug(signs, 0.5), Gaussian(0.05), RandomMask(0.1)])

        # contrast_transforms = transforms.RandomChoice([Gaussian(0.1), RandomMask(0.2)])

        train_data = Gene_data(X_train, y_train,transform=ContrastiveLearningViewGenerator(contrast_transforms))
        # train_data = Gene_data(X_train, y_train,transform=contrast_transforms)
        test_data = Gene_data(X_val, y_val, transform=None)

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    for study in range(6):
        print(f"test {names[i]}")
        train_loader = train_loaders[study]
        val_loader = val_loaders[study]
        test_loader = data.DataLoader(all_data[study], batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        model = SimCLR(input_dim,emb=feat_dim, out_dim=proj_dim).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4, last_epoch=-1)
        # loss_func = SupConLoss(temperature= tmp)
        loss_func = losses.SubCenterArcFaceLoss(num_classes=7, embedding_size=64).to(device)
        runname = f'{names[study]}-RC-AllAug-TentAll'
        trainer = Trainer(model, train_loader, val_loader, test_loader, loss_func, optimizer, scheduler, max_epochs, batch_size, 
                            run_name=runname)
        trainer.train()
        torch.save(trainer.model.state_dict(), f'models/simclr{study}.pt')
        classify = classifier(feat_dim).to('cuda')
        trainer.linear_probe(model, classify, study, 300)
        # trainer.test()
        trainer.tent()

