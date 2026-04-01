# normalization.py
# HW2, Computational Genomics, Spring 2025
# andrewid: tnair

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np
import matplotlib.pyplot as plt

PER_MILLION = 1/1000000
PER_KILOBASE = 1/1000

# Do not change this function signature
def rpkm(raw_counts, gene_lengths):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    ## 
    n_genes, n_samples = raw_counts.shape
    total_reads = np.sum(raw_counts, axis=0)
    
    rpkm_matrix = np.zeros_like(raw_counts, dtype=float)
    for i in range(n_genes):
        for j in range(n_samples):
            rpkm_matrix[i, j] = (raw_counts[i, j] * 1e9) / (total_reads[j] * gene_lengths[i])
    return rpkm_matrix
   
# define any helper function here    


# Do not change this function signature
def size_factor(raw_counts):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    n_genes, n_samples = raw_counts.shape
    geo_means = np.zeros(n_genes)
    for i in range(n_genes):
        non_zero = raw_counts[i, :][raw_counts[i, :] > 0]
        if len(non_zero) > 0:
            geo_means[i] = np.exp(np.mean(np.log(non_zero)))
    
    # Calculate size factors for each sample
    size_factors = np.zeros(n_samples)
    for j in range(n_samples):
        ratios = []
        for i in range(n_genes):
            if geo_means[i] > 0:
                ratios.append(raw_counts[i, j] / geo_means[i])
        size_factors[j] = np.median(ratios)
    
    # Normalize
    normalized = raw_counts / size_factors
    return normalized
    

if __name__=="__main__":
    raw_counts=np.loadtxt(sys.argv[1])
    gene_lengths=np.loadtxt(sys.argv[2])
    
    rpkm1=rpkm(raw_counts, gene_lengths)
    size_factor1=size_factor(raw_counts)

    # TODO: write plotting code here
    rpkm_counts = rpkm(raw_counts, gene_lengths)
    size_factor_counts = size_factor(raw_counts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].boxplot(np.log2(raw_counts + 1))
    axes[0].set_title('Raw Counts')
    
    axes[1].boxplot(np.log2(rpkm1 + 1))
    axes[1].set_title('RPKM Normalized')
    
    axes[2].boxplot(np.log2(size_factor1 + 1))
    axes[2].set_title('Size Factor Normalized')
    
    plt.tight_layout()
    plt.savefig('normalization_boxplots.png')
    plt.show()
    np.savetxt('size_factor_normalized_counts.txt', size_factor1)