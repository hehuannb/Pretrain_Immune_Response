import numpy as np
from scipy.stats import rankdata, pointbiserialr
import torch
import pandas as pd
# Define the gene signature as a list of genes
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve,f1_score
from sklearn.linear_model import LogisticRegression



# Define function to compute per-sample gene signature score using biserial correlation coefficient
def compute_per_sample_gene_signature_score(expression_df, gene_set, class_df):
    """Compute per-sample gene signature score using biserial correlation coefficient."""
    assert len(expression_df) == len(class_df), "Expression and class DataFrames must have the same length."

    gene_set_expression_df = expression_df.loc[:, gene_set]

    gene_scores = []
    for i in range(len(expression_df)):
        corr_coeffs, _ = pointbiserialr(gene_set_expression_df.iloc[i,:], class_df.iloc[i,:])
        gene_scores.append(np.sum(corr_coeffs))

    return np.array(gene_scores)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

allen_x, allen_y = torch.load('preprocessed/ax.pt'), torch.load('preprocessed/ay.pt')
gide_x, gide_y =  torch.load('preprocessed/gx.pt'), torch.load('preprocessed/gy.pt')
liu_x, liu_y =  torch.load('preprocessed/lx.pt'), torch.load('preprocessed/ly.pt')
riaz_x, riaz_y=  torch.load('preprocessed/rx.pt'), torch.load('preprocessed/ry.pt')
hugo_x, hugo_y=  torch.load('preprocessed/hx.pt'), torch.load('preprocessed/hy.pt')
lee_x, lee_y=  torch.load('preprocessed/leex.pt'), torch.load('preprocessed/leey.pt')

allen_x = np.log(allen_x+1)
allen_x=(allen_x-allen_x.mean())/allen_x.std()
allen_x.reset_index(drop=True, inplace=True)

gide_x = np.log(gide_x+1)
gide_x=(gide_x-gide_x.mean())/gide_x.std()
gide_x.reset_index(drop=True, inplace=True)

liu_x = np.log(liu_x+1)
liu_x=(liu_x-liu_x.mean())/liu_x.std()
liu_x.reset_index(drop=True, inplace=True)

riaz_x = np.log(riaz_x+1)
riaz_x=(riaz_x-riaz_x.mean())/riaz_x.std()
riaz_x.reset_index(drop=True, inplace=True)

hugo_x = np.log(hugo_x+1)
hugo_x=(hugo_x-hugo_x.mean())/hugo_x.std()
hugo_x.reset_index(drop=True, inplace=True)

lee_x = np.log(lee_x+1)
lee_x=(lee_x-lee_x.mean())/lee_x.std()
lee_x.reset_index(drop=True, inplace=True)

names = ['allen','riaz','gide','liu','hugo','lee']
all_data = [allen_x, riaz_x, gide_x,liu_x,hugo_x,lee_x]
all_labels = [allen_y, riaz_y, gide_y,liu_y,hugo_y,lee_y]
# Define a set of genes for the gene signature
gene_set =["CXCR6","CD2","IFNG", "ITGAL",\
           "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11","PDCD1",\
            "GZMB","GZMK", "IDO1", "PRF1", "NKG7","PRF1","GZMA", "HLA-DRA"]
for i in range(6):
    print(f"test {names[i]}")
    train = all_data[:i] + all_data[i+1:]
    label = all_labels[:i] + all_labels[i+1:]
    df = pd.concat(train)
    labels = pd.concat(label)
    gene_set_expression_df = df[gene_set]
    X_train, X_test, y_train, y_test = train_test_split(gene_set_expression_df.values, \
                                                        labels.values, test_size=0.3)

    gene_scores = []
    for g in range(X_train.shape[1]):
        corr_coeffs, _ = pointbiserialr(X_train[:,g], y_train)
        gene_scores.append(np.sum(corr_coeffs))
    print(gene_scores)
    rowR = np.where(y_train == 1)
    rowNR = np.where(y_train == 0)
    x = X_train@np.array(gene_scores)
    x = NormalizeData(x)
    scoreR = x[rowR]
    scoreNR = x[rowNR]
    fig, ax = plt.subplots()

    # Create a box plot of the two arrays
    bp = ax.boxplot([scoreR, scoreNR], labels=['R', 'NR'])
    # Set axis labels and title
    ax.set_xlabel('Score')
    ax.set_ylabel('Value')
    ax.set_title('Boxplot of scores (Train)')
    plt.show()


    dx, dy = all_data[i][gene_set].values, all_labels[i].values
    rowR = np.where(dy == 1)
    rowNR = np.where(dy == 0)
    ds = dx@np.array(gene_scores)
    ds = NormalizeData(ds)
    scoreRG = ds[rowR]
    scoreNRG = ds[rowNR]
    fig, ax = plt.subplots()

    # Create a box plot of the two arrays
    bp = ax.boxplot([scoreRG, scoreNRG], labels=['R', 'NR'])
    # Set axis labels and title
    ax.set_xlabel('Score')
    ax.set_ylabel('Label')
    ax.set_title(f'Boxplot of scores {names[i]}')

    # Show the plot
    plt.show()


    best_acc = 0
    best_f1 = 0
    best_threshold = 0
    best_auc = 0
    X_val = X_train@np.array(gene_scores)
    X_val = NormalizeData(X_val)
    for threshold in np.arange(0,1,50):
        y_pred = X_val >= threshold
        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        auc = roc_auc_score(y_train,y_pred)
        if acc > best_acc:
            best_acc = acc
            # best_threshold = threshold
        if f1 > best_f1:
            best_f1 = f1
            # best_threshold = threshold
        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold
    # Print the best threshold and performance metrics
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best accuracy: {1-best_acc:.3f}")
    print(f"Best F1 score: {best_f1:.3f}")
    print(f"Best AUC score: {best_auc:.3f}")


    y_d = ds>=best_threshold
    acc = accuracy_score(dy, y_d)
    f1 = f1_score(dy, y_d)
    auc = roc_auc_score(dy,y_d)
    print(1-acc, f1, auc)
