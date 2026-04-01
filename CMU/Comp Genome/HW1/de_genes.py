# de_genes.py
# HW2, Computational Genomics, Spring 2025
# andrewid: tnair

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np


# Do not change this function signature

def bh(genes, pvals, alpha):
    """(list, list, float) -> numpy array
    applies benjamini-hochberg procedure
    
    Parameters
    ----------
    genes: name of genes 
    pvalues: corresponding pvals
    alpha: desired false discovery rate
    
    Returns
    -------
    array containing gene names of significant genes.
    gene names do not need to be in any specific order.
    """
    ## Thought Process:
    # 1. Create a dictionary mapping genes to p-values
    # 2. Sort the dictionary by p-values
    # 3. Iterate through the sorted list; apply BH condition
    arr = []
    zipped = dict(zip(genes, pvals))
    zipped = sorted(zipped.items(), key=lambda item: item[1])
    arr = []
    for i in range(len(genes)):
        if pvals[i] <= (alpha * (i + 1)) / len(pvals):
            arr.append(genes[i])
    return np.array(arr)

# define any helper function here    

if __name__=="__main__":
    # Here is a free test case
    genes=['a', 'b', 'c']
    input1 = [0.01, 0.04, 0.1]
    print(bh(genes, input1, 0.05))
