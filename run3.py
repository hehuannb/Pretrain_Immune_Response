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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
from pytorch_metric_learning import losses
from scipy import stats
import matplotlib.pyplot as plt
from utils import load_data, gene_signatures
from torch.distributions import Beta

def report(tmp, lr, epoch, train_l,val_l, mode='lp',i=0):
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    model.load_state_dict(torch.load(f'models/simclr{i}.pt'))
    classifier = Net(emb, tmp).to(device)
    criterion = nn.CrossEntropyLoss()
    if mode=='lp':
        model.projection = nn.Identity()
        # optimizer = optim.Adam(classifier.parameters(), lr=lr)
        optimizer = optim.LBFGS(classifier.parameters(), history_size=10, max_iter=6,line_search_fn="strong_wolfe")
        model.eval()
    elif mode=='ft':
        model.projection = nn.Identity()
        model.train()
        # optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),\
                            #    lr=lr)
        optimizer = optim.LBFGS(list(model.parameters()) + list(classifier.parameters()), \
                                history_size=10, max_iter=6,line_search_fn="strong_wolfe")
    else:
        raise('not implemented')
    best_f1 = 0
    for _ in range(epoch):
        model.train()
        classifier.train()
        batch, labels = train_l.dataset.x_train, train_l.dataset.y_train
        labels = labels.type(torch.LongTensor)
        batch, labels = batch.to(device), labels.to(device)# on GPU
        def closure():
            optimizer.zero_grad()
            if mode == 'lp':
                model.eval()
                with torch.no_grad():
                    feats = model(batch)
                outputs = classifier(feats.detach())
            elif mode=='ft':
                feats = model(batch)
                outputs = classifier(feats)
            else:
                raise('not implemented')
            loss = criterion(outputs, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        #### Eval
        model.eval()
        classifier.eval()
        batch, labels = val_l.dataset.x_train, val_l.dataset.y_train
        labels = labels.type(torch.LongTensor)
        batch, labels = batch.to(device), labels.to(device)# on GPU
        if mode == 'lp':
            feats = model(batch)
            outputs = classifier(feats.detach())
        elif mode=='ft':
            feats = model(batch)
            outputs = classifier(feats)
        pred = torch.argmax(outputs, dim=1)
        labels = labels.cpu()
        pred = pred.cpu()
        score = outputs[:,1].detach().cpu()
        roc_auc = roc_auc_score(labels, score)
        val_f1 = f1_score(labels, pred)
        if roc_auc > best_f1:
            best_f1 = roc_auc
            torch.save(classifier.state_dict(), f'models/best_c{i}.pth')
            torch.save(model.state_dict(), f'models/best_m{i}.pth')
            bmodel = model
            bclass = classifier
    print(f"Val: Acc={accuracy_score(labels, pred):.3f}, F1={val_f1:.3f}, auc={roc_auc:.3f}")
    return bmodel, bclass

def deploy(model, classifier, loader,mode='ft'):
    model.eval()
    classifier.eval()
    data_, target_ = loader.dataset.x_train, loader.dataset.y_train
    data_, target_ = data_.to(device), target_.to(device)# on GPU
    if mode == 'lp':
        feats = model(data_)
        outputs = classifier(feats)
    elif mode=='ft':
        feats = model(data_)
        outputs = classifier(feats)
    else:
        raise('not implemented')
    _,pred = torch.max(outputs, dim=1)
    target_ = target_.cpu()
    pred = pred.cpu()
    f1 = f1_score(target_, pred)
    roc_auc = roc_auc_score(target_, outputs[:,1].detach().cpu())
    print(f"Deploy:Acc={accuracy_score(target_, pred):.3f}, F1={f1:.3f}, auc={roc_auc:.3f}")
    return target_, pred, outputs.detach().cpu()
    

def vis(i, genes):
    features_all = {}
    labels_all = {}
    allen_data, riaz_data, gide_data,  liu_data, hugo_data ,lee_data = load_data(genes, None)

    fg, lg = prepare_data_features(model, gide_data,i)
    fa, la = prepare_data_features(model, allen_data,i)
    fl, ll = prepare_data_features(model, liu_data,i)
    fr, lriaz = prepare_data_features(model, riaz_data,i)
    fh, lh = prepare_data_features(model, hugo_data,i)
    flee, llee = prepare_data_features(model, lee_data,i)
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

    mapper = umap.UMAP(n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        metric='euclidean',
        random_state=42).fit(fea.to('cpu'))
    umap.plot.points(mapper,labels=np.array(l1),theme='fire')
    umap.plot.points(mapper,labels=np.array(l2),theme='fire')


@torch.no_grad()
def prepare_data_features(model, dataset,i):
    # Prepare model
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    # model.projection = nn.Identity()
    model.load_state_dict(torch.load(f'models/simclr{i}.pt'))    
    network = deepcopy(model.backbone)
    network.train()
    network.to('cuda')

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=31, num_workers=1, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to('cuda')
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)


    return feats, labels


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, emb),
                                    nn.BatchNorm1d(emb),
                                    nn.ReLU(inplace=True),
                                    # nn.Linear(hidden,emb),
                                    # nn.BatchNorm1d(emb),
                                    # nn.ReLU(inplace=True),
          )
    def forward(self,x):
        return self.encoder(x)
    
class SimCLR(nn.Module):
    def __init__(self, hidden, emb, input_dim):
        super().__init__()
        self.backbone = Encoder(input_dim, hidden=hidden, emb=emb)
        self.projection = nn.Sequential(
            nn.Linear(in_features=emb, out_features=32),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        return self.projection(embedding)
    
class Net(nn.Module):
    def __init__(self,input_shape, temp):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32,2)
        # self.n1 = nn.Linear(input_shape,2)
        # self.tmp = temp
        self.softmax = nn.Softmax(dim=1)
        self.do1 = nn.Dropout(0.3)  

    def forward(self,x):
        x = self.do1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return self.softmax(x)

class domain_aug(object):
    def __init__(self, genes, study=0):
        allen_data, riaz_data, gide_data, liu_data, hugo_data, lee_data = load_data(genes, None)
        all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]
        train = all_data[:study] + all_data[study+1:]
        self.d = [train[i].x_train for i in range(5)]
        self.y = [train[i].y_train for i in range(5)]
        self.r = [np.where(self.y[i]==1)[0] for i in range(5)]
        self.nr = [np.where(self.y[i]==0)[0] for i in range(5)]

    def __call__(self, x, y):
        lam1 = torch.rand(5)
        lam1 /= torch.sum(lam1)
        if y ==1:
            m1 = [self.d[i][np.random.choice(self.r[i])] * lam1[i] for i in range(5)]
        else:
            m1 = [self.d[i][np.random.choice(self.nr[i])] * lam1[i] for i in range(5)]
        mix = sum(m1)
        return 0.7 * x + 0.3 * mix

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Gene_data(Dataset):
    def __init__(self,X_train, y_train, transform=None):
        if torch.is_tensor(X_train):
            x = X_train.clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            # self.x_train = torch.log(x+1)
            self.y_train=torch.tensor(y_train,dtype=torch.int32)
        else:
            x = torch.tensor(X_train.values,dtype=torch.float32).clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            # self.x_train = torch.log(x+1)
            self.y_train= torch.tensor(y_train.values,dtype=torch.int32)
        self.transform = transform
        self.beta =  Beta(torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))
        self.ridx = np.where(self.y_train==1)[0]
        self.nridx = np.where(self.y_train==0)[0]
        self.test = 0
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        if self.transform is not None:
            # xgn = self.transform[0](x)
            xgn = self.transform[0](x,y)
            sig_idx = self.transform[1]
            if y ==1:
                mixup_idx = np.random.choice(self.ridx)
            else:
                mixup_idx = np.random.choice(self.nridx)
            lam = self.beta.sample() * torch.ones(len(x))
            lam[sig_idx] = 1
            xsn = lam * x + (1-lam) * self.x_train[mixup_idx]
            x = [xgn, xsn]
        return x, y
    
def deploy2(lr, epochs,mode, study):
    train_l = train_loaders[study]
    test_l = test_loaders[study]
    
    model, classifier = report(0.2, lr,epochs,train_l,test_l,mode,study)
    deploy_loader = data.DataLoader(all_data[study], batch_size=batch_size, shuffle=True,
                                drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    target, pred, outputs = deploy(model, classifier,deploy_loader,mode)
    # vis(i)
    acc = accuracy_score(target, pred)
    f1 = f1_score(target, pred)
    cm = confusion_matrix(target, pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(xticks_rotation=45)
    disp.ax_.set_title(f"Acc={acc:.2f}, F1={f1:.2f}",fontsize = 20)
    disp.im_.colorbar.remove()
    for labels in disp.text_.ravel():
        labels.set_fontsize(30)

    plt.show()   
    fpr, tpr, thresholds = roc_curve(target, outputs[:,1])
    auc = roc_auc_score(target,outputs[:,1])
    plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (auc))
    plt.title(f"test  {names[study]}",fontsize = 20)
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc="lower right",fontsize = 20)
    plt.show()

if __name__ == "__main__":
    cudnn.deterministic = True
    cudnn.benchmark = True
    NUM_WORKERS = 1
    emb = 128
    hidden = 512
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 500
    device='cuda'
    tmp = 0.05
    batch_size = 64 
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
    metrics = []
    models = []
   
    names = ['allen','riaz','gide','liu','hugo','lee']
    train_loaders = []
    test_loaders = []
    for i in range(6):
        print(f"test {names[i]}")
        train = all_data[:i] + all_data[i+1:]
        X = torch.cat([x.x_train for x in train])
        y = torch.cat([x.y_train for x in train])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        input_dim = X_train.shape[1]
        contrast_transforms = [domain_aug(genes, i), signs]
        train_data = Gene_data(X_train, y_train,transform=contrast_transforms)
        test_data = Gene_data(X_test, y_test, transform=contrast_transforms)

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    for study in range(6):
        train_loader = train_loaders[study]
        test_loader = test_loaders[study]
        model = SimCLR(hidden, emb, input_dim).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4,
                                                            last_epoch=-1)
        # loss_func = SupConLoss(temperature= tmp)
        loss_func = losses.SubCenterArcFaceLoss(num_classes=7, embedding_size=64).to(device)
        for epoch_counter in tqdm(range(max_epochs)):
            # for (batch1,labels1) in zip(cycle(train_loader), tcga_loader):
            for _, (batch1,labels1) in enumerate(train_loader):
                batch = torch.cat(batch1, dim=0).to('cuda')
                labels1 = labels1.type(torch.LongTensor).to('cuda')
                feats = model(batch)
                f1,f2 = torch.chunk(feats,2, dim=0)
                features = torch.cat([f1, f2], dim=1)
                # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_func(features, labels1) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            if epoch_counter % 25==0:
                print(loss.item())

        torch.save(model.state_dict(), f'models/simclr{study}.pt')
        # deploy2(5e-4, 300,'lp', study)
        # vis(study, genes)
    for i in range(6):
        deploy2(1e-3, 400,'ft', i)
