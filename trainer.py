import wandb
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
import torch.optim as optim
import tent
import matplotlib.pyplot as plt
import copy

# Define your model architecture here
class Encoder(nn.Module):
    def __init__(self, input_dim, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, emb),
                                    nn.BatchNorm1d(emb),
                                    nn.ReLU(inplace=True),
          )
        # self.tt = nn.TransformerEncoderLayer(emb, 4)
        # self.encoder = nn.TransformerEncoder(self.tt, 1)
    def forward(self,x):
        x = self.encoder(x)
        # x, _ = torch.max(output, dim=1)
        return x
    
class SimCLR(nn.Module):
    def __init__(self,input_dim, emb = 128, out_dim = 32):
        super().__init__()
        self.backbone = Encoder(input_dim, emb=emb)
        self.projection = nn.Sequential(
            nn.Linear(in_features=emb, out_features=out_dim),
        )
    def forward(self, x):
        embedding = self.backbone(x)
        return F.normalize(self.projection(embedding),p=2,dim=1)
    
class classifier(nn.Module):
    def __init__(self,input_dim=32):
        super(classifier,self).__init__()
        self.fc1 = nn.Linear(input_dim,2)
        self.bn1 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(32,2)
        self.softmax = nn.Softmax(dim=1)
        self.do1 = nn.Dropout(0.3)  

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.fc2(x)
        # x = self.do1(self.fc1(x))
        # return self.softmax(x)
        return F.log_softmax(x, dim=1)

class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss

class Trainer:
    def __init__(self, model, train_data, val_data, test_data, criterion, optimizer,scheduler, num_epochs, batch_size,\
                  device="cuda", run_name="your-run-name"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.scheduler = scheduler
        self.coral = CORAL()
        # Initialize wandb
        # wandb.init(project='immune response', entity='hehuannb', name=run_name, save_code=False)

        # # Log model architecture to wandb
        # wandb.watch(model)

    def train(self):
        for epoch_counter in range(self.num_epochs):
            train_loss = self._train_step()
            # if epoch_counter % 25 == 0:
            print(train_loss)
            # Log metrics to wandb
            # wandb.log({"train_loss": train_loss})
       

    def _train_step(self):
        self.model.train()
        train_loss = 0.0
        for _, (batch1, labels1) in enumerate(self.train_data):
            batch = torch.cat(batch1, dim=0).to('cuda')
            labels1 = labels1.type(torch.LongTensor).to('cuda')
            feats = self.model(batch)
            f1,f2 = torch.chunk(feats,2, dim=0)
            # features = torch.cat([f1, f2], dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            loss = self.criterion(features,labels1) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()
        return train_loss / len(self.train_data)

    # def fine_tune(self, i, tcga_loader):
    #     self.model.train()
    #     for epoch_counter in tqdm(range(self.num_epochs)):
    #         train_loss = 0.0
    #         joint_loaders = enumerate(zip(self.train_data, tcga_loader))
    #         for _, ((batch1,labels1),(batch2,labels2)) in joint_loaders:
    #             batch = torch.cat(batch1, dim=0).to('cuda')
    #             labels1 = labels1.type(torch.LongTensor).to('cuda')
    #             feats = self.model(batch)
    #             f1,f2 = torch.chunk(feats,2, dim=0)
    #             features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

    #             batch = torch.cat(batch2, dim=0).to('cuda')
    #             feats2 = self.model(batch)
    #             cons = 10 * self.coral(feats2, feats)
    #             loss = self.criterion(features, labels1) + cons
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             train_loss += loss.item()
    #         if epoch_counter % 25 == 0:
    #             print(train_loss / len(self.train_data), cons.item())
    #     torch.save(self.model.state_dict(), f'models/simclr{i}.pt')

    def fine_tune(self, i, tcga_loader):
        self.model.train()
        for epoch_counter in tqdm(range(self.num_epochs)):
            train_loss = 0.0
            for _, (batch1,labels1) in enumerate(self.train_data):
                batch = torch.cat(batch1, dim=0).to('cuda')
                labels1 = labels1.type(torch.LongTensor).to('cuda')
                feats = self.model(batch)
                f1,f2 = torch.chunk(feats,2, dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = self.criterion(features, labels1) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()
            if epoch_counter % 50 == 0:
                print(train_loss / len(self.train_data))
        torch.save(self.model.state_dict(), f'models/simclr{i}.pt')

    def linear_probe(self, classifier,i, epoch):
        # self.model.load_state_dict(torch.load(f'models/simclr_tcga2.pt'))
        self.model.load_state_dict(torch.load(f'models/simclr{i}.pt'))
        self.model.projection = nn.Identity()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        # optimizer = optim.LBFGS(classifier.parameters(), history_size=10, max_iter=6,line_search_fn="strong_wolfe")
        self.model.eval()
        best_auc = 0
        best_f1 = 0
        for epoch_counter in range(epoch):
            classifier.train()
            batch, labels = self.train_data.dataset.x_train, self.train_data.dataset.y_train
            labels = labels.type(torch.LongTensor)
            batch, labels = batch.to(self.device), labels.to(self.device)# on GPU
            # def closure():
            #     optimizer.zero_grad()
            #     with torch.no_grad():
            #         feats = self.model(batch)
            #     outputs = classifier(feats.detach())
            #     loss = criterion(outputs, labels)
            #     loss.backward()
            #     return loss
            # optimizer.step(closure)
            optimizer.zero_grad()
            with torch.no_grad():
                feats = self.model(batch)
            outputs = classifier(feats.detach())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            classifier.eval()
            batch, labels = self.val_data.dataset.x_train, self.val_data.dataset.y_train
            labels = labels.type(torch.LongTensor)
            batch, labels = batch.to(self.device), labels.to(self.device)# on GPU
            with torch.no_grad():
                feats = self.model(batch)
                outputs = classifier(feats)
            pred = torch.argmax(outputs, dim=1)
            labels = labels.cpu()
            pred = pred.cpu()
            score = outputs[:,1].detach().cpu()
            roc_auc = roc_auc_score(labels, score)
            val_f1 = f1_score(labels, pred)
            val_acc = accuracy_score(labels, pred)
            if val_f1 >= best_f1:
                best_auc = roc_auc
                best_f1 = val_f1
                best_acc = val_acc
                self.bmodel = self.model
                self.bclass = classifier
                fpr, tpr, thresholds = roc_curve(labels, score)
            if epoch_counter % 50 == 0:
                print(best_f1)
            # wandb.log({"In study Test Acc": val_acc, "In study Test F1": val_f1, "In study Test AUC": roc_auc})
        print(f"Val: Acc={best_acc:.3f}, F1={best_f1:.3f}, auc={best_auc:.3f}")
        # wandb.log({"Val Acc": best_acc, "Val F1": best_f1, "Val AUC": best_auc})
        plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (best_auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.legend(loc="lower right",fontsize = 20)
        plt.show()
        return best_acc, best_f1, best_auc

    def test(self):
        bmodel = copy.deepcopy(self.bmodel)
        bclass = copy.deepcopy(self.bclass)
        bmodel.eval()
        bclass.eval()
        data_, target_ = self.test_data.dataset.x_train, self.test_data.dataset.y_train
        data_, target_ = data_.to(self.device), target_.to(self.device)# on GPU
        with torch.no_grad():
            feats = bmodel(data_)
            outputs = bclass(feats)
        _, pred = torch.max(outputs, dim=1)
        target_ = target_.cpu()
        pred = pred.cpu()
        f1 = f1_score(target_, pred)
        score = outputs.detach().cpu()
        roc_auc = roc_auc_score(target_,score[:,1])
        acc = accuracy_score(target_, pred)
        print(f"Deploy:Acc={acc:.3f}, F1={f1:.3f}, auc={roc_auc:.3f}")
        # fpr, tpr, thresholds = roc_curve(target_, score[:,1])
        # plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (roc_auc))
        # plt.plot([0, 1], [0, 1],'r--')
        # plt.legend(loc="lower right",fontsize = 20)
        # plt.show()
        # wandb.log({"Cross Study Test Acc": acc, "Cross Study Test F1": f1, "Cross Study Test AUC": roc_auc})
        # wandb.log({"ROC Curve" : wandb.plot.roc_curve(target_,score,labels=["NR", "R"])})
        # wandb.finish()
        return acc, f1, roc_auc

    def tent(self):
        bmodel = copy.deepcopy(self.bmodel)
        bclass = copy.deepcopy(self.bclass)
        bmodel.eval()
        bclass.eval()
        bmodel.projection = bclass
        data_, target_ = self.test_data.dataset.x_train, self.test_data.dataset.y_train
        data_, target_ = data_.to(self.device), target_.to(self.device)# on GPU
        # with torch.no_grad():
        #     feats = self.bmodel(data_)
        model = tent.configure_model(bmodel)
        params, param_names = tent.collect_params(model)
        optimizer = optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(model, optimizer, steps=10)
        outputs = tented_model(data_)  # now it infers and adapts!
        _, pred = torch.max(outputs, dim=1)
        target_ = target_.cpu()
        pred = pred.cpu()
        f1 = f1_score(target_, pred)
        score = outputs.detach().cpu()
        roc_auc = roc_auc_score(target_,score[:,1])
        acc = accuracy_score(target_, pred)
        print(f"Deploy:Acc={acc:.3f}, F1={f1:.3f}, auc={roc_auc:.3f}")
        fpr, tpr, thresholds = roc_curve(target_, score[:,1])
        plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.legend(loc="lower right",fontsize = 20)
        plt.show()
        # wandb.log({"Cross Study Test Acc": acc, "Cross Study Test F1": f1, "Cross Study Test AUC": roc_auc})
        # wandb.log({"ROC Curve" : wandb.plot.roc_curve(target_,score,labels=["NR", "R"])})
        # wandb.finish()