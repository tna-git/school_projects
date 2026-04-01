# # 3a

# # Submit a log2-log2 scatterplot of dispersion vs mean in the size factor counts dataset, for the
# # two conditions (but with all the data on one plot). Use different colors for case and control
# # condition. How would you interpret this plot, and why?
import numpy as np
import matplotlib.pyplot as plt

size_factor_counts = np.loadtxt('size_factor_normalized_counts.txt')
labels = np.loadtxt('labels.txt')
gene_names = np.loadtxt('GeneNames.txt', dtype=str)

cases = size_factor_counts[:, (labels == 1)]
controls = size_factor_counts[:, (labels == 2)]

case_mu = np.mean(cases, axis=1)
case_std = np.std(cases, axis=1)
case_dispersion = (case_std ** 2) / case_mu

control_mu = np.mean(controls, axis=1)
control_std = np.std(controls, axis=1)
control_dispersion = (control_std ** 2) / control_mu


plt.figure(figsize=(10, 6))
plt.scatter(np.log2(case_mu), np.log2(case_dispersion), color='red', label='Case', alpha=0.5, s = 20)
plt.scatter(np.log2(control_mu), np.log2(control_dispersion), color='green', label='Control', alpha=0.5, s = 20)
plt.xlabel('log2(Mean Expression)')
plt.ylabel('log2(Dispersion)')
plt.title('Dispersion vs Mean Expression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3a_ dispersion_vs_mean.png', dpi=300)
plt.show()

# 3b
# Compute the log2 fold change of the genes in the two conditions. Select top 10
# upregulated genes, and top 10 downregulated genes, and submit a gene expression heatmap for
# the selected genes. Place both upregulated and dowregulated genes on the same heatmap
import pandas as pd
import seaborn as sns

df = pd.DataFrame(size_factor_counts,index=gene_names,columns=labels.astype(int))
#print(df.head(5))
case_mean = df.loc[:, 1].mean(axis=1)
control_mean = df.loc[:, 2].mean(axis=1)
log2_fc = np.log2((case_mean + 1) / (control_mean + 1))
df['log2_FC'] = log2_fc
df_sorted = df.sort_values(by='log2_FC', ascending=False)
top_upregulated = df_sorted.head(10)
top_downregulated = df_sorted.tail(10)
# print(df_sorted.head(10))
# print(df_sorted.tail(10))

selected_genes = pd.concat([top_upregulated, top_downregulated])
selected_genes = selected_genes.drop(columns=['log2_FC'])

plt.figure(figsize=(8, 6))
sns.heatmap(np.log2(selected_genes), cmap='coolwarm', cbar_kws={'label': 'log2(counts)'})

# 3. Add title and display the plot
plt.title("Heatmap of DataFrame Values")
plt.xlabel('Control or Case Sample')
plt.ylabel('Genes')
plt.title('Top 10 Upregulated and Top 10 Downregulated Genes')
plt.tight_layout()
plt.savefig('3b_de_genes_heatmap.png', dpi=500)
plt.show()

## 4C i. Conversion Code
import scanpy as sc
import pandas as pd
import numpy as np
adata = sc.read_h5ad('sc_counts.h5ad')

# Convert to dense if sparse
if hasattr(adata.X, 'toarray'):
    counts = adata.X.toarray()
else:
    counts = adata.X
df = pd.DataFrame(counts.T, 
                  index=adata.var_names,
                  columns=adata.obs_names)

df.to_csv('counts_matrix.csv')
print(f"Saved: {df.shape[0]} genes x {df.shape[1]} cells")