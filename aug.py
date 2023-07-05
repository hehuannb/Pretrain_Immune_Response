import random
import torch 
import numpy as np
from utils import load_data
from torch.distributions import Beta

class domain_aug(object):
    def __init__(self, genes, study=0):
        allen_data, riaz_data, gide_data, liu_data, hugo_data, lee_data = load_data(genes, None)
        all_data = [allen_data, riaz_data, gide_data, liu_data, hugo_data,lee_data]
        train = all_data[:study] + all_data[study+1:]
        self.d = [train[i].x_train for i in range(5)]
        self.y = [train[i].y_train for i in range(5)]
        self.r = [np.where(self.y[i]==1)[0] for i in range(5)]
        self.nr = [np.where(self.y[i]==0)[0] for i in range(5)]

    def __call__(self, x, y,x2):
        # lam1 = torch.rand(5)
        # lam1 /= torch.sum(lam1)
        lam1 = torch.tensor(np.random.dirichlet(np.ones(5),size=1))[0]
        if y ==1:
            m1 = [self.d[i][np.random.choice(self.r[i])] * 0.2 for i in range(5)]
        else:
            m1 = [self.d[i][np.random.choice(self.nr[i])] *0.2 for i in range(5)]
        mix = sum(m1)
        return 0.6 * x + 0.4 * mix

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class tcga_mixup(object):
    def __init__(self, tcga, beta):
        self.t = tcga.x_train
        self.l = self.t.shape[0]
        self.beta =  Beta(torch.FloatTensor([beta]), torch.FloatTensor([beta]))

    def __call__(self, x, y, x2):
        b1 = self.beta.sample()
        m1 = self.t[np.random.choice(self.l)]
        return b1* x + (1-b1) * m1

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class normal_mixup(object):
    def __init__(self, normal, beta):
        self.t = normal.x_train
        self.l = self.t.shape[0]
        self.beta =  Beta(torch.FloatTensor([beta]), torch.FloatTensor([beta]))

    def __call__(self, x, y, x2):
        b1 = self.beta.sample()
        m1 = self.t[np.random.choice(self.l)]
        return b1* x + (1-b1) * m1

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class Gaussian(object):
    def __init__(self, std=0.05):
        self.p = std
    
    def __call__(self, X, y,x2):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            Gaussian Noise
        """
        return X + torch.normal(mean=0, std=self.p, size=X.size())

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomMask(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, X, y, x2):
        
        bit_mask = torch.FloatTensor(X.shape).uniform_() > self.p
        return X * bit_mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class FixedSigAug(object):
    def __init__(self, sig, beta):
        self.sig_idx = sig
        self.beta =  Beta(torch.FloatTensor([beta]), torch.FloatTensor([beta]))
        
    def __call__(self, x, y, x2):
        lam = self.beta.sample() * torch.ones(len(x))
        lam[self.sig_idx] = 1
        xsn = lam * x + (1-lam) * x2
        return xsn

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class RandomSigAug(object):
    def __init__(self, sig, beta):
        self.sig_idx = sig
        self.beta =  Beta(torch.FloatTensor([beta]), torch.FloatTensor([beta]))
        
    def __call__(self, x, y, x2):
        lam = self.beta.sample() * torch.ones(len(x))
        ss = np.random.choice(self.sig_idx, 150, replace=False)
        lam[ss] = 1
        xsn = lam * x + (1-lam) * x2
        return xsn

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


