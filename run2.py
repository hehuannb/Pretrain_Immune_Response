
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
from aug import Gaussian, RandomMask, mixup
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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
from pytorch_metric_learning import losses, testers
from scipy import stats
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import StandardScaler

def report(tmp, lr, epoch, train_l,val_l, mode='lp',i=0):
    model = SimCLR(hidden,emb, input_dim).to('cuda')
    model.load_state_dict(torch.load(f'models/simclr{i}.pt'))
    classifier = Net(emb, tmp).to(device)
    criterion = nn.CrossEntropyLoss()
    if mode=='lp':
        model.projection = nn.Identity()
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
        model.eval()
    elif mode=='ft':
        model.projection = nn.Identity()
        model.train()
        optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()),\
                               lr=lr, momentum=0.9)
    else:
        raise('not implemented')
    best_f1 = 0
    for _ in range(epoch):
        model.train()
        classifier.train()
        batch, labels = train_l.dataset.x_train, train_l.dataset.y_train
        labels = labels.type(torch.LongTensor)
        batch, labels = batch.to(device), labels.to(device)# on GPU
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
        optimizer.step()
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
        roc_auc = roc_auc_score(labels, pred)
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
    roc_auc = roc_auc_score(target_, pred)
    # cm = confusion_matrix(target_.cpu(), pred.cpu())
    # ConfusionMatrixDisplay(cm).plot()
    print(f"Deploy:Acc={accuracy_score(target_, pred):.3f}, F1={f1:.3f}, auc={roc_auc:.3f}")
    # RocCurveDisplay.from_predictions( target_.cpu(), pred.cpu())
    return target_, pred
    
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def vis(i):
    features_all = {}
    labels_all = {}
    allen_data = Gene_data(allen_x, allen_y)
    riaz_data = Gene_data(riaz_x, riaz_y)
    gide_data = Gene_data(gide_x, gide_y)
    liu_data = Gene_data(liu_x, liu_y)
    hugo_data = Gene_data(hugo_x, hugo_y)
    lee_data = Gene_data(lee_x, lee_y)
    tcga_data = Gene_data(tcga,pd.DataFrame(np.ones(tcga.shape[0])))
    luad_data = Gene_data(luad,pd.DataFrame(np.ones(luad.shape[0])))
    fg, lg = prepare_data_features(model, gide_data,i)
    fa, la = prepare_data_features(model, allen_data,i)
    fl, ll = prepare_data_features(model, liu_data,i)
    fr, lriaz = prepare_data_features(model, riaz_data,i)
    fh, lh = prepare_data_features(model, hugo_data,i)
    flee, llee = prepare_data_features(model, lee_data,i)
    ft,_ = prepare_data_features(model, tcga_data, i)
    fluad,_ = prepare_data_features(model, luad_data, i)
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
    features_all['tcga'] = ft
    features_all['luad'] = fluad
    fea = torch.vstack([fa, fg, fr, fh, fl,flee, ft, fluad])
    l1 = [str(la[i].item()) for i in range(len(fa))] + [str(lg[i].item()) for i in range(len(fg))]\
         + [str(lriaz[i].item()) for i in range(len(fr))] \
        +[str(lh[i].item()) for i in range(len(fh))] +[str(ll[i].item()) for i in range(len(fl))] \
        + [str(llee[i].item()) for i in range(len(flee))] + ['skcm' for i in range(len(ft))]\
        + ['luad' for i in range(len(fluad))]

    l2 = ['allen:' for i in range(len(fa))] + ['gide:' for i in range(len(fg))]\
         + ['riaz:' for i in range(len(fr))]\
        +['hugo:' for i in range(len(fh))] + ['liu:' for i in range(len(fl))]\
        + ['lee:' for i in range(len(flee))]  + ['skcm:' for i in range(len(ft))]+ ['luad:' for i in range(len(fluad))]

    mapper = umap.UMAP(n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        metric='euclidean',
        random_state=42).fit(fea.to('cpu'))
    umap.plot.points(mapper,labels=np.array(l1),theme='fire')
    umap.plot.points(mapper,labels=np.array(l2),theme='fire')




class Gene_data(Dataset):
 
    def __init__(self,X_train, y_train, transform=None):
        if torch.is_tensor(X_train):
            x = X_train.clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            self.y_train=torch.tensor(y_train,dtype=torch.int32)
        else:
            x = torch.tensor(X_train.values,dtype=torch.float32).clone().detach()
            self.x_train = normalize(torch.log(x+1), p=2.0, dim = 0)
            self.y_train= torch.tensor(y_train.values,dtype=torch.int32)
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        x = self.x_train[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y_train[idx]

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

    # Sort images by labels
    # labels, idxs = labels.sort()
    # feats = feats[idxs]

    return feats, labels


@torch.no_grad()
def pseduo_label(f1, f2, contrast_feature, labels1, topk):
    fp, fn = contrast_feature[torch.where(labels1==1)], contrast_feature[torch.where(labels1==0)]
    fpc, fnc = fp.mean(0), fn.mean(0)
    distP = F.cosine_similarity(f1,fpc)
    distP2 = F.cosine_similarity(f2,fpc)
    dpm = (distP+distP2)/2
    distN = F.cosine_similarity(f1,fnc)
    distN2 = F.cosine_similarity(f2,fnc)
    dnm = (distN+distN2)/2
    index_sortedP = torch.argsort(dpm)
    index_sortedN = torch.argsort(dnm)
    top_idx_p = index_sortedP[:topk]
    top_idx_n = index_sortedN[:topk]
    pl = torch.cat([torch.ones(len(top_idx_p)), torch.zeros(len(top_idx_n))])
    fea_aug1 = torch.cat([f1[top_idx_p], \
                        f1[top_idx_n]], dim=0)
    fea_aug2 = torch.cat([f2[top_idx_p], \
                        f2[top_idx_n]], dim=0)

    return torch.cat([fea_aug1.unsqueeze(1),fea_aug2.unsqueeze(1)], dim=1), pl.to('cuda')



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden),
                                    nn.BatchNorm1d(hidden),
                                    nn.ReLU(inplace=True),
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
    
class Net(nn.Module):
    def __init__(self,input_shape, temp):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16,2)
        self.n1 = nn.Linear(input_shape,2)
        self.tmp = temp
        self.softmax = nn.Softmax(dim=1)
        self.do1 = nn.Dropout(0.2)  # 20% Probability

    def forward(self,x):
        x = self.do1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return self.softmax(x)


if __name__ == "__main__":
    cudnn.deterministic = True
    cudnn.benchmark = True
    NUM_WORKERS = 1
    emb = 64
    hidden = 512
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 200
    device='cuda'
    tmp = 0.2
    batch_size = 64 
    X, y = torch.load('preprocessed/X.pt'), torch.load('preprocessed/y.pt')
    tcga = torch.load('tcga_skcm.pt')
    luad = torch.load('tcga_luda.pt')
    duplicate_cols = tcga.columns[tcga.columns.duplicated()]
    tcga.drop(columns=duplicate_cols, inplace=True)
    # immune_gene = pd.read_csv('/home/huan/Downloads/InnateDB_genes.csv')
    # immunegene = X.columns.intersection(immune_gene.name)
    # X = X[immunegene]
    # mad = []
    # for i in range(X.shape[1]):
    #     x = X.iloc[:, i].values
    #     mad.append(stats.median_abs_deviation(x))
    # mad = np.array(mad)
    # top5k = mad.argsort()[-5000:]
    # gtex =  pd.read_csv("gene_tpm_2017-06-05_v8_skin_not_sun_exposed_suprapubic.gct", sep='\t',skiprows=2)
    # gtex.drop(['id','Name'],1,inplace=True)
    # gtex = gtex.set_index('Description')
    # gtex = gtex.T
    # cc = gtex.columns.intersection(X.columns)
    # gg = gtex[cc]
    t5 = np.loadtxt('top10k.txt')
    gg = torch.load('gtex.pt')
    X = X.iloc[:, t5]
    cc =  X.columns.intersection(gg.columns)
    cc = cc.intersection(tcga.columns)
    X = X[cc] 
    gg = gg[cc]
    normal_tissue = torch.tensor(gg.values,dtype=torch.float32)
    normal_tissue = normalize(torch.log(normal_tissue+1), p=2.0, dim = 0)
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
    tcga = tcga[genes]
    luad = luad[genes]
    contrast_transforms = transforms.RandomChoice([Gaussian(std=0.1), mixup(normal_tissue)])
    # contrast_transforms = transforms.RandomChoice([Gaussian(std=0.1), RandomMask(p=0.1)])
    allen_data = Gene_data(allen_x, allen_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    riaz_data = Gene_data(riaz_x, riaz_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    gide_data = Gene_data(gide_x, gide_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    liu_data = Gene_data(liu_x, liu_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    hugo_data = Gene_data(hugo_x, hugo_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    lee_data = Gene_data(lee_x, lee_y, transform=ContrastiveLearningViewGenerator(contrast_transforms))
    normal_data = Gene_data(normal_tissue, torch.ones(normal_tissue.shape[0]))
    tcga_data = Gene_data(tcga,pd.DataFrame(np.ones(tcga.shape[0])),\
                          transform=ContrastiveLearningViewGenerator(contrast_transforms))
    luad_data = Gene_data(luad,pd.DataFrame(np.ones(luad.shape[0])),\
                          transform=ContrastiveLearningViewGenerator(contrast_transforms))
    gide_loader = data.DataLoader(gide_data, batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    riaz_loader = data.DataLoader(riaz_data, batch_size=32, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    allen_loader = data.DataLoader(allen_data, batch_size=32, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    hugo_loader = data.DataLoader(hugo_data, batch_size=13, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    lee_loader = data.DataLoader(lee_data, batch_size=22, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)  
    liu_loader = data.DataLoader(liu_data, batch_size=batch_size, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    
    normal_loader = data.DataLoader(normal_data, batch_size=batch_size, shuffle=True,
                                drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    tcga_loader = data.DataLoader(tcga_data, batch_size=batch_size, shuffle=True,
                                drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    luad_loader = data.DataLoader(luad_data, batch_size=batch_size, shuffle=True,
                                drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    
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
        train_data = Gene_data(X_train, y_train,transform=ContrastiveLearningViewGenerator(contrast_transforms))
        test_data = Gene_data(X_test, y_test, transform=ContrastiveLearningViewGenerator(contrast_transforms))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    for i in range(2,4):
        train_loader = train_loaders[i]
        test_loader = test_loaders[i]
        model = SimCLR(hidden, emb, input_dim).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len((train_loader)), eta_min=0,
                                                            last_epoch=-1)
        loss_func = SupConLoss(temperature= tmp)
        all_iter  = iter(luad_loader)
        # loss_func = losses.SubCenterArcFaceLoss(num_classes=7, embedding_size=128).to(device)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        for epoch_counter in tqdm(range(max_epochs)):
            for (batch1,labels1), (batchT,_) in zip(cycle(train_loader), tcga_loader):
            # for _, (batch1,labels1) in enumerate(train_loader):
                batch = torch.cat(batch1, dim=0).to('cuda')
                labels1 = labels1.type(torch.LongTensor).to('cuda')
                feats = model(batch)
                f1,f2 = torch.chunk(feats,2, dim=0)
                # features = torch.cat([f1, f2], dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss1 = loss_func(features, labels1) 
                # align
                batch_a = torch.cat(batchT, dim=0).to('cuda')
                feats = model(batch_a)
                f1_u,f2_u = torch.chunk(feats,2, dim=0)
                contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
                feat_pl, pl = pseduo_label(f1_u, f2_u, contrast_feature, labels1,16)
                loss2 = loss_func(feat_pl,pl)
                try:
                    batch_luad, _  = next(all_iter)
                except StopIteration:
                    all_iter  = iter(luad_loader)
                batch_l = torch.cat(batch_luad, dim=0).to('cuda')
                feats = model(batch_l)
                f1_l,f2_l = torch.chunk(feats,2, dim=0)
                # features = torch.cat([f1, f2], dim=1)
                # features = torch.cat([f1_l.unsqueeze(1), f2_l.unsqueeze(1)], dim=1)
                
                loss3 = triplet_loss(f1, f2, f1_l)+triplet_loss(f1, f2, f2_l)

                loss = 1 * loss1 + 1*loss3+0.3* loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()



            if epoch_counter % 25==0:
                print(loss1.item())

        torch.save(model.state_dict(), f'models/simclr{i}.pt')

    for j in range(1):
        # fig, axs = plt.subplots(2, 2,  figsize=(12, 12))
        for i in range(2,4):
            train_l = train_loaders[i]
            test_l = test_loaders[i]
            fig, axs = plt.subplots(2, 1,  figsize=(6, 12))
            model, classifier = report(0.2, 1e-3,300,train_l,test_l,'lp',i)
            deploy_loader = data.DataLoader(all_data[i], batch_size=batch_size, shuffle=True,
                                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
            target, pred = deploy(model, classifier,deploy_loader,'lp')
            # vis(i)
            acc = accuracy_score(target, pred)
            f1 = f1_score(target, pred)
            cm = confusion_matrix(target, pred)
            disp = ConfusionMatrixDisplay(cm)
            fpr, tpr, thresholds = roc_curve(target, pred)
            auc = roc_auc_score(target,pred)
            axs[1].plot(fpr, tpr, label='ROC (area = %0.2f)' % (auc))
            axs[1].set_title(f"test  {names[i]}",fontsize = 20)
            # Custom settings for the plot 
            axs[1].plot([0, 1], [0, 1],'r--')
            axs[1].legend(loc="lower right",fontsize = 20)
            disp.plot(ax=axs[0], xticks_rotation=45)
            disp.ax_.set_title(f"Acc={acc:.2f}, F1={f1:.2f}",fontsize = 20)
            disp.im_.colorbar.remove()
            for labels in disp.text_.ravel():
                labels.set_fontsize(30)

            plt.show()   # Display

