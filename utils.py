import os
import shutil
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.utils.data import Dataset
import numpy as np

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
    
def load_data(genes, aug):
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
    allen_data = Gene_data(allen_x, allen_y, transform=aug)
    riaz_data = Gene_data(riaz_x, riaz_y, transform=aug)
    gide_data = Gene_data(gide_x, gide_y, transform=aug)
    liu_data = Gene_data(liu_x, liu_y, transform=aug)
    hugo_data = Gene_data(hugo_x, hugo_y, transform=aug)
    lee_data = Gene_data(lee_x, lee_y, transform=aug)

    return allen_data, riaz_data, gide_data, liu_data, hugo_data, lee_data


def info_nce_loss(args, features):

    labels = torch.cat([torch.arange(args.batch) for i in range(args.nviews)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temp
    return logits, labels


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

# def report_metric(pred, label):
def gene_signatures(genes):
    sets = ['TRGC2', 'GZMH', 'KLRD1', 'FGFBP2', 'GZMB', 'PRF1', 'SAMD3', 'MATK', 'GNLY', 'NKG7', \
            'GZMA','PDCD1', 'HAVCR2', 'LAG3', 'ENTPD1', 'CD38', 'TOX', \
            'FOXP3', 'CCR8', 'PMCH', 'CCR4', 'RTKN2', 'CTLA4', \
            'CD19', 'MS4A1', 'TNFRSF13C', 'VPREB3', 'PAX5', 'CR2',\
            'CSF3R', 'MS4A6A', 'MS4A7', 'MNDA', 'C5AR1', 'FCGR2A', 'C3AR1', 'FPR1', \
            'LILRB2', 'HDC', 'FCGR3B', 'CCL22',\
            'IDO1', 'CXCL10', 'CXCL9', 'HLA-DRA', 'STAT1', 'IFNG',\
            'DTYMK', 'SKA2', 'SAE1', 'RFC2', 'GMPS', 'TMEM185A', 'MKI67', 'PCNA', 'STMN1', 'HMGB2',\
            'CD3D', 'CD3G', 'TRAC', 'BCL11B', 'TRAT1', 'CD2','CD8B',\
            'GNLY', 'GZMA', 'GZMB', 'GZMH', 'KLRB1', 'KLRD1', 'KLRK1', 'PRF1', 'NKG7',\
            'NKG7', 'CD8A', 'GZMH', 'CD8B', 'HCST', 'CST7', 'KLRD1', 'GZMA', 'CTSW', 'PRF1', \
            'GZMB', 'GZMM', 'KLRK1', 'HLA-C', 'KLRC4', 'AOAH', 'GNLY', 'CCL4', 'MATK', 'CCL4L1', \
            'ZNF683', 'ABI3','EOMES', 'LAG3', 'PTGER4','CD38', 'CTLA4', 'ENTPD1', 'EPSTI1', 'FABP5', \
            'HAVCR2', 'NDUFB3', 'PDCD1', 'PRDX3', 'SIRPG', 'SNAP47', 'SNRPD1', 'UBE2F', 'WARS1', \
            'CXCL13', 'AFAP1L2', 'TNFRSF9', 'MYO7A', 'GOLIM4',\
            'IL2RA', 'IKZF2','DERL3', 'FCRL2', 'FCRL5', 'IGLL5', 'TNFRSF17',\
            'LILRA4', 'LRRC26', 'SCT', 'ASIP', 'SLC12A3', 'PTCRA',\
            'LIM2', 'KIR2DL4', 'KLRC1', 'IL18RAP', 'KLRF1',\
           'DCN', 'PCOLCE', 'EMILIN1', 'CYGB', 'MFAP4', 'TCF21', 'BGN', 'ACTA2',\
            'ECSCR', 'CCL14', 'KDR', 'TIE1', 'PCAT19', 'MYCT1', 'FLT4',\
            'KLK1', 'SMIM22', 'LGALS4', 'CLDN7', 'CDH1', 'HMGCS2',\
            'CTSG', 'FCGR3B', 'CLC', 'HDC','CLEC4D', 'FCAR', 'MCEMP1',\
             'GPBAR1', 'CDH23', 'LIPN', 'MS4A14',\
            'MSR1', 'CD163', 'TREM2', 'LILRB5', 'SDS', 'MRC1', 'SLCO2B1',\
            'CPA3', 'HPGDS', 'VWA5A','UPK3A', 'CLEC9A', 'CCL22', 'ZNF366', 'CD1C',
         'TYR', 'SLC45A2', 'CDH19', 'PMEL', 'SLC24A5', 'MAGEA6', 'GJB1', 'PLP1', 'PRAME', \
            'CAPN3', 'ERBB3', 'GPM6B', 'S100B', 'PAX3', 'S100A1', 'MLANA', 'ZNF207', 'HNRNPK', 'SNRPD3', 'SRRM1', 'ACTB',
         'GNLY', 'FGFBP2', 'CX3CR1', 'KLF2', 'TBX21',
         'PDCD1', 'LAG3', 'CXCL13', 'RBPJ', 'ZBED2', 'ETV1', 'ID3', 'MAF', 'PRDM1', 'EOMES', 'IFNG',
         'TNFRSF1B', 'RGS2', 'TIGIT', 'CD27', 'TNFRSF9', 'SLA', 'RNF19A', 'INPP5F', 'XCL2', 'HLA-DMA', \
            'FAM3C', 'UQCRC1', 'WARS1', 'EIF3L', 'KCNK5', 'TMBIM6', 'CD200', 'ZC3H7A', 'SH2D1A', 'ATP1B3', 'MYO7A', 'THADA',
         'TRAT1', 'CD40LG', 'LEF1', 'TNFRSF4', 'IL17RE', 'RORC', 'IL23R','HNRNPU', 'GZMH', 'TIGIT', 'CXCL13', 'CD4', 'IL17A',\
           'FOXP3', 'IL2RA', 'IL10RA', 'IKZF2', 'RTKN2', 'CDC25B', 'S1PR4',
         'FOXP3', 'CCR8', 'TNFRSF18', 'LAYN', 'IKZF2', 'RTKN2', 'CTLA4', 'BATF', 'IL21R',
         'GPR171', 'PKIA', 'AP3M2', 'SERINC5', 'PHACTR2', 'CD84', 'GZMK', 'IL7R',
         'TRGC1', 'TRGC2', 'TRDC', 'TRDV1', 'TRDV2','CCR7', 'IL7R', 'TCF7','TCF7', 'LEF1', 'NELL2', 'CAMK4', 'MAL', 'TRABD2A',
         'IL7R', 'PLAC8', 'SELL', 'ADD3', 'TNFAIP8', 'TCF7','TNFRSF13B', 'AIM2', 'IGHG1', 'IGHG2', 'CLECL1', 'IGHA1',
         'IGHD', 'IGHM', 'CD72', 'TCL1A', 'FCER2', 'BTLA', 'FCRL1','LST1', 'LTB', 'PRMT9', 'ALDOC','NCR1', \
            'XCL2', 'XCL1', 'IL21R', 'KIR2DL3', 'KIR3DL1', 'KIR3DL2','MARCO', 'CXCL5', 'SLAMF9', 'MMP19',
         'TREM2', 'MRC1', 'CD209', 'PLA2G7', 'SLCO2B1', 'SIGLEC1', 'FOLR2',
         'CXCL10', 'CXCL11', 'CXCL9','CD300E', 'SLC11A1', 'CLEC12A', 'FCN1', 'LILRA5', 'LYPD2', 'S100A12',
         'P2RY12', 'SIGLEC8', 'TMEM119','CD163', 'CD68', 'CD84', 'MS4A4A', 'CCL13', 'CD209', 'HSD11B1',
         'CLEC9A', 'XCR1', 'XCR1', 'CLNK', 'ENPP1', 'PPM1J', 'ZNF366','ENHO', 'CD1C', 'CLEC10A', 'CD1E',
         'CCL22', 'CCL17', 'CCL19', 'HMSD', 'NCCRP1', 'UBD', 'CRLF2',
         'ELANE', 'MPO', 'PRTN3', 'CTSG', 'AZU1', 'FCGR3B','CLC', 'MS4A3', 'HDC', 'CPA3', 'IL4',
         'CLC', 'PRG2', 'EPX', 'RNASE3', 'RNASE2', 'CD24', 'SIGLEC8', 'FUT9', 'CCR3','MS4A2', 'TPSAB1', 'CPA3', 'HDC', 'TPSB2',
         'CSF3R', 'S100A12', 'CEACAM3', 'FCAR', 'FCGR3B', 'FPR1', 'SIGLEC5',
         'AVP', 'CRHBP', 'CD34', 'NRIP1',
         'ALDOB', 'APOH', 'PCK1', 'APOC3', 'TAT',
         'PAN3', 'KCNN4', 'ERMAP', 'TPM1', 'HBB', 'AHSP', 'CA1', 'HBD',
         'GCG', 'GC', 'IRX2', 'CRYBA2', 'INS', 'IAPP', 'HADH', 'ADCYAP1', 'MAFA', 'LEPR', 'RGS2', 'RBP4', 'BAIAP3', 'PPY', 'STMN2', 'DPYSL3', 'FGB', 'MEIS1',
         'KRT7', 'SOX9', 'EPCAM', 'KRT19', 'TFF1', 'CLDN4', 'MMP7', 'TFF2', 'SFRP5', 'SLC12A2', 'CITED4', 'FGFR2', 'CD24',
         'LEP', 'FABP4', 'TCF21', 'ADIPOQ', 'P2RX5', 'UCP1', 'CIDEA', 'PRDM16', 'LHX8',
         'SOD3', 'CSPG4', 'RGS5', 'NDUFA4L2', 'COX4I2',
         'S100P', 'CEACAM6', 'DPEP1', 'MAL2', 'REG4',
         'ABCA3', 'PGC', 'AGER', 'NAPSA', 'SPTB', 'CFTR',
         'CCDC80', 'UPK3B', 'PLA2G2A', 'PTGIS', 'PRG4', 'CALB2',
         'ACRBP','ANLN', 'AURKA', 'AURKB', 'BIRC5', 'CDC20', 'CDCA2', 'CDCA3', 'CDK1', 'CENPA', 'CENPE', 'KIF11', 'CD3D', 'CD3E', 'CD3G', \
            'CD6', 'SH2D1A', 'TRAT1', 'IL17A', 'PTPN13', 'RORC', 'IL17F', 'ITPR1', 'CXCL13', 'PKIA', 'TNFRSF4', 'AP3M2', 'AQP3', 'STING1', \
        'SERINC5', 'CD40LG', 'CXCR6', 'SLC4A10', 'IL7R', 'IL18RAP', 'LGALS3', 'ZBTB16', 'CXXC5', 'NCR3', 'CEBPD', 'SLC4A10', 'CCR2', 'S1PR5', \
         'KLRG1', 'SLAMF6', 'CXCR3', 'S1PR1', 'ITGB7', 'CD8A', 'IL7R', 'TCF7', 'PDCD1', 'BLK', 'CD19', 'MS4A1', 'TNFRSF17', 'FCRL2', 'FAM30A',\
         'PNOC', 'SPIB', 'TCL1A', 'CD160', 'SPON2', 'FCGR3A', 'CCL4', 'FGFBP2', 'S1PR5', 'FCRL6', 'XCL1', 'SELL', 'CCR7', 'FUT7', \
         'KLRC1', 'TNFRSF18', 'CCL3', 'SOD2', 'IL1B', 'IER3', 'CAMP', 'AXL', 'OTULINL', 'NBPF10', 'LILRB4', 'SIGLEC6', 'IL11', \
        'CHI3L1', 'WNT2', 'CEMIP', 'COL7A1', 'MMP3', 'CCL19', 'CCL21', 'CXCL13', 'CCR7', 'CXCR5', 'SELL', 'LAMP3', 'CD79B', 'CD1D', \
        'LAT', 'SKAP1', 'CETP', 'EIF1AY', 'RBP5', 'PTGDS', 'BZW2', 'CCT3', 'CDK4', 'GPATCH4', 'ISYNA1', 'MDH2', 'PPIA', 'RPL31', 'RPL37A',\
        'RPL41', 'RPS21', 'AHNAK', 'APOD', 'ATP1A1', 'B2M', 'CD44', 'CD63', 'CTSB', 'CTSD', 'FOS', 'GRN', 'HLA-A', 'UBC', 'ACTB', 'HNRNPK',\
              'YWHAZ', 'ABCF1', 'ACTB', 'ALAS1', 'B2M', 'CLTC', 'G6PD', 'GAPDH', 'GUSB', 'HPRT1', 'LDHA', 'PGK1', 'C1orf43', 'CHMP2A', \
                'EMC7', 'GPI', 'PSMB2', 'PSMB4', 'RAB7A', 'REEP5', 'SNRPD3']

 
    sets = list(set(sets))
    indexs = []
    for i in range(len(sets)):
        loc = np.where(genes.values==sets[i]) 
        if len(loc[0])!=0:
            indexs.append(loc[0])
    print(len(indexs))
    return np.array(indexs)