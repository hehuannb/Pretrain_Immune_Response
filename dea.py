import numpy as np
import pandas as pd
import umap
import scanpy as sc
import anndata as ad
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('all_data.csv',index_col=0)

var = pd.DataFrame(index=df.index)


labels = np.loadtxt('cluster_label.txt')
newL = []
for i in (labels):
    if i==-1:
        newL.append('mix')
    if i==0:
        newL.append('mix')
    if i==1:
        newL.append('NR')
    if i==2:
        newL.append('NR')
    if i==3:
        newL.append('mix')
    if i==4:
        newL.append('R')

renres = pd.read_csv('res.csv', index_col=0)
renres[renres.response==1] = 'non-response'
renres[renres.response==0] = 'response'
obs = pd.DataFrame()
obs['gene'] = df.columns
obs['genea'] = df.columns
obs.set_index('gene', inplace=True)




var_names = df.index
var = pd.DataFrame(index=var_names)
var['cluster'] = pd.Categorical(newL).astype(str)

var['label'] = pd.Categorical(renres.response.values)
adata = ad.AnnData(df.values, obs=var, var=obs, dtype='float32')



astrocyte_marker = 'IFNG'
embeddings = np.loadtxt("embed.txt")
cluster2 = adata[adata.obs['cluster']=='R']
not_cluster2 = adata[adata.obs['cluster'] =='NR'] 
cluster2_marker_exp = cluster2[:,cluster2.var['gene']==astrocyte_marker] 
not_cluster2_marker_exp = not_cluster2[:,cluster2.var['gene']==astrocyte_marker] 
fig, ax = plt.subplots()
sns.kdeplot(cluster2_marker_exp.to_df().values.squeeze(), ax=ax, color='red', label='Cluster:R')
sns.kdeplot(not_cluster2_marker_exp.to_df().values.squeeze(), ax=ax, color='green',label='Cluster:NR')
plt.legend()



combined_dfs = pd.DataFrame({'Response': cluster2_marker_exp.to_df().values.squeeze(),
                             'Non-Response': not_cluster2_marker_exp.to_df().values.squeeze()[0:90]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()


fig, ax = plt.subplots()
cluster2 = adata[adata.obs['label']=='response']
not_cluster2 = adata[adata.obs['label'] =='non-response'] 
astrocyte_marker = 'IFNG'
cluster2_marker_exp = cluster2[:,cluster2.var_names==astrocyte_marker] 
not_cluster2_marker_exp = not_cluster2[:,cluster2.var_names==astrocyte_marker] 
sns.kdeplot(cluster2_marker_exp.to_df().values.squeeze(), ax=ax, color='red', label='Label:R')
sns.kdeplot(not_cluster2_marker_exp.to_df().values.squeeze(), ax=ax, color='green',label='Label:NR')
plt.legend()




from scipy.stats import ttest_ind

ttest = ttest_ind(cluster2_marker_exp, 
          not_cluster2_marker_exp, 
          equal_var=False, # it's not necessarily fair to assume that these two populations have equal variance
          nan_policy='omit') # omit NaN values
print(ttest)

sc.tl.rank_genes_groups(adata, 'cluster', n_genes=10,method='logreg')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
sc.tl.rank_genes_groups(adata, 'cluster',n_genes=10, method='t-test')
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, log=True,groupby='cluster',dendrogram=True) # plot the result
# 
# sc.pl.matrixplot(adata, n_genes=20, groupby='response') # plot the result


marker_genes = {
'IFN': ['IFNG', 'STAT1', 'CCR5', 'CXCL9', 'CXCL10', \
        'CXCL11', 'IDO1', 'PRF1', 'GZMA'],}

# sl = []
# for i in ['IFNG', 'STAT1', 'CCR5', 'CXCL9', 'CXCL10',  'CXCL11', 'IDO1', 'PRF1', 'GZMA']:
#     s = obs[obs.gene==i].index.values
#     sl.append(s)
# marker_genes = {
# 'IFN gamma signature': sl,}
mm = {
'logreg': ['TMSB4X',
 'SERPINE2',
 'LYZ',
 'RPL37A',
 'RPL4',
 'CTSB',
 'RPS19',
 'RPS24',
 'APOD',
 'VIM'],}

mt = {
'ttest':['SNURF',
 'TMEM244',
 'INO80B',
 'NBPF20',
 'GLRA1',
 'PCDHA8',
 'PTCD1',
 'COX20',
 'STON1-GTF2A1L',
 'DEFB134']

}

sc.pl.matrixplot(adata, mm, groupby='cluster', use_raw=False)
sc.pl.matrixplot(adata, mt, groupby='cluster', use_raw=False)

# with rc_context({'figure.figsize': (4.5, 3)}):
#     sc.pl.violin(adata, ['STAT1', 'CXCL9'], groupby='cluster',stripplot=False)


# sc.pl.matrixplot(adata, marker_genes, 'cluster', dendrogram=True,
#                  colorbar_title='mean z-score', vmin=-2, vmax=2, cmap='RdBu_r')


sc.pl.rank_genes_groups_dotplot(adata, n_genes=4)

ax = sc.pl.matrixplot(adata, marker_genes, groupby='cluster',  \
                   vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=True, swap_axes=True, figsize=(11,4))

sc.pl.rank_genes_groups_violin(adata,'cluster', n_genes=20, jitter=False)