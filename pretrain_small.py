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
import pandas as pd
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
from aug import domain_aug, Gaussian,RandomMask,FixedSigAug, RandomSigAug, normal_mixup
from losses import SupConLoss
import seaborn as sns
import os, umap

def word_to_num_func(word):
    return word_to_num.get(word, 0)

def num_to_word_func(num):
    return num2word.get(num, 'unknown')

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
    feat_dim = 64
    proj_dim = 32
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 500
    device='cuda'
    tmp = 0.2
    batch_size = 64 

    # Load Datasets
    X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')
    tcga = torch.load('tcga_x.pt')
    duplicate_cols = tcga.columns[tcga.columns.duplicated()]
    tcga.drop(columns=duplicate_cols, inplace=True)
    t5 = np.loadtxt('top10k.txt')
    X = X.iloc[:, t5]
    cc =  X.columns.intersection(tcga.columns)
    X = X[cc] 
    genes = X.columns
    tcga = tcga[genes]
    normal = torch.load('tcga_normal.pt')
    normal = normal[genes]
    signs = gene_signatures(genes).squeeze()
    
    #### Reduce feature space
    tcga = tcga.iloc[:,signs]
    normal = normal.iloc[:,signs]
    
    cases = torch.load('tcga_y.pt')

    b, c = np.unique(cases, return_inverse=True)
    word_to_num = {b[i]: i for i in range(len(b))}
    num2word = {i: b[i] for i in range(len(b))}
    arr_nums = np.vectorize(word_to_num_func)(cases)


    train_normal = Gene_data(normal, pd.Series(c),None)
    contrast_transforms = transforms.RandomChoice([normal_mixup(train_normal, 0.7), Gaussian(0.05)])
    train_tcga = Gene_data(tcga,pd.Series(np.zeros(len(normal))) ,transform=ContrastiveLearningViewGenerator(contrast_transforms))
    tcga_loader = data.DataLoader(train_tcga, batch_size=512, shuffle=True,
                                    drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    input_dim = 272
    # model = SimCLR(input_dim,emb=feat_dim, out_dim=proj_dim).to('cuda')
    # optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4, last_epoch=-1)
    # loss_func = SupConLoss(temperature=tmp)
    # runname = 'TCGA-Pretrain'
    # trainer = Trainer(model, tcga_loader, None, None, loss_func, optimizer, scheduler, max_epochs, batch_size, 
    #                     run_name=runname)
    # trainer.train()
    # torch.save(trainer.model.state_dict(), f'models/simclr_tcga2.pt')

    

    allen_data, riaz_data, gide_data,  liu_data, hugo_data ,lee_data = load_data(genes, None)
    all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]
    for item in all_data:
        item.x_train = item.x_train[:,signs]

    names = ['allen','riaz','gide','liu','hugo','lee']
    train_loaders = []
    val_loaders = []
    accs = []
    fs = []
    aucs = []
    vaccs = []
    vfs = []
    vaucs = []
    for i in range(6):
        train = all_data[:i] + all_data[i+1:]
        X = torch.cat([x.x_train for x in train])
        y = torch.cat([x.y_train for x in train])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        input_dim = 272
        contrast_transforms = transforms.RandomChoice([normal_mixup(train_normal, 0.7), Gaussian(0.05)])
        train_data = Gene_data(X_train, y_train,transform=ContrastiveLearningViewGenerator(contrast_transforms))
        test_data = Gene_data(X_val, y_val, transform=None)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        # x_test = all_data[i]
        # x_test.x_train = x_test.x_train[:,signs]
        test_loader = data.DataLoader(all_data[i], batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        model = SimCLR(input_dim,emb=feat_dim, out_dim=proj_dim).to('cuda')
        model.load_state_dict(torch.load('models/simclr_tcga2.pt'))
        optimizer = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4, last_epoch=-1)
        loss_func = SupConLoss(temperature= tmp)

        runname = f'{names[i]}-RC-AllAug-TentAll'
        trainer = Trainer(model, train_loader, val_loader, test_loader, loss_func, optimizer, \
                          scheduler, 200, batch_size, 
                            run_name=runname)

        trainer.fine_tune(i, tcga_loader)
        classify = classifier(feat_dim).to('cuda')
        va, vf, vauc = trainer.linear_probe(classify,i, 300)
        a, f, auc = trainer.test()
        accs.append(a)
        fs.append(f)
        aucs.append(auc)
        vaccs.append(va)
        vfs.append(vf)
        vaucs.append(vauc)
    print("In:", np.mean(vaccs), np.mean(vfs), np.mean(vaucs))
    print("Cross:", np.mean(accs), np.mean(fs), np.mean(aucs))
        # trainer.tent()

    # za = []
    # yl = []
    # model = SimCLR(input_dim,emb=feat_dim, out_dim=proj_dim).to('cuda')
    # model.load_state_dict(torch.load('models/simclr1.pt'))
    # model.eval()
    # model.projection =nn.Identity()
    # train_data = Gene_data(tcga, pd.Series(arr_nums.squeeze()),transform=None)
    # train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
    #                                 drop_last=True, pin_memory=True, \
    #                                     num_workers=NUM_WORKERS)
    # for i, (batch, l) in enumerate(train_loader):
    #     batch = batch.to(device)
    #     f = model(batch)
    #     za.append(f.cpu().detach().numpy())
    #     yl.append(l)
    # za = np.vstack(za)
    # yl = np.concatenate(yl)
    # y2string = np.vectorize(num_to_word_func)(yl)
    # yll = np.zeros(len(yl),)-1
    # print(za.shape[0])



    # brain_umap = umap.UMAP(random_state=999, n_neighbors=32, min_dist=1)
    # embedding = pd.DataFrame(brain_umap.fit_transform(za), columns = ['UMAP1','UMAP2'])
    # palette = sns.color_palette("bright", 36)  #Choosing color
    # sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
    #                 hue=y2string, palette=palette,
    #                 alpha=.9, linewidth=0, s=2, legend='full')

    # sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5), ncol=3)
    # new_list = np.array(['TCGA' if x !='TCGA-SKCM' else x for x in y2string])

    # allen_data, riaz_data, gide_data,  liu_data, hugo_data ,lee_data = load_data(genes, None)
    # all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]
    # names = ['allen','riaz','gide','liu','hugo','lee']

    # for i in range(6):
    #     X = all_data[i].x_train
    #     y = all_data[i].y_train
    #     print(y.shape)
    #     train_data = Gene_data(X, y,transform=None)
    #     train_loader = data.DataLoader(train_data, batch_size=12, shuffle=True,
    #                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    #     for _, (batch, l) in enumerate(train_loader):
    #         batch = batch.to(device)
    #         f = model(batch)
    #         za = np.vstack([za,f.cpu().detach().numpy()])
    #         yll = np.concatenate([yll, l])
    #         new_list= np.concatenate([new_list, np.repeat(names[i],repeats=len(l))])

    # brain_umap = umap.UMAP(random_state=999, n_neighbors=32, min_dist=1)
    # embedding = pd.DataFrame(brain_umap.fit_transform(za), columns = ['UMAP1','UMAP2'])
    # palette = sns.color_palette("bright", 8)  #Choosing color
    # sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
    #                 hue=new_list, palette=palette,
    #                 alpha=.9, linewidth=0, s=2, legend='full')

    # sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5), ncol=1)