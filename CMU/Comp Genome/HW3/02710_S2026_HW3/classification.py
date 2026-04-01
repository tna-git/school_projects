# classification.py
# HW3, Computational Genomics, Spring 2026
# andrewid: tnair

# WARNING: Do not change the file name; Autograder expects it.

import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def get_top_gene_filter(data, n_keep=2000):
    """Select top n_keep most dispersed genes.

    Args:
        data (n x m matrix): input gene expression data of shape num_cells x num_genes
        n_keep (int): number of genes to keep after filtration; default 2000

    Returns:
        filter (array of length n_keep): an array of column indices that can be used as an
            index to keep only certain genes in data. Each element of filter is the column
            index of a highly-dispersed gene in data.
    """
    means = np.mean(data, axis=0)
    means = np.where(means != 0, means, 1e-6)
    vars_ = np.var(data, axis=0)
    disps = vars_ / means
    idx = disps.argsort()[-n_keep:]
    return idx


def reduce_dimensionality_pca(
    filtered_train_gene_expression,
    filtered_test_gene_expression,
    n_components=20
):
    """Train a PCA model and use it to reduce the training and testing data.

    Args:
        filtered_train_gene_expression (n_train x num_top_genes matrix): input filtered training expression data
        filtered_test_gene_expression (n_test x num_top_genes matrix): input filtered test expression data

    Return:
        (reduced_train_data, reduced_test_data): a tuple of
            1. The filtered training data transformed to the PC space.
            2. The filtered test data transformed to the PC space.
    """
    n_train = filtered_train_gene_expression.shape[0]
    n_test = filtered_test_gene_expression.shape[0]
    all_data = np.vstack([filtered_train_gene_expression, filtered_test_gene_expression])
    pca = PCA(n_components=n_components)
    all_reduced = pca.fit_transform(all_data)
    reduced_train_data = all_reduced[:n_train, :]
    reduced_test_data = all_reduced[n_train:n_train + n_test, :]
    return (reduced_train_data, reduced_test_data)


def plot_transformed_cells(reduced_train_data, train_labels):
    """Plot the PCA-reduced training data using just the first 2 principal components.

    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        train_labels (array of length n_train): array of cell type labels for training data

    Return:
        None
    """
    mini_df = pd.DataFrame({"PC1": reduced_train_data[:, 0],"PC2": reduced_train_data[:, 1],"label": train_labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=mini_df,x="PC1",y="PC2",hue="label",palette="tab10",s=20,edgecolor="none")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of training cells")
    plt.legend(title="Cell type", loc="upper left")
    plt.tight_layout()
    #plt.savefig("q2.png")
    plt.show()


def train_and_evaluate_svm_classifier(
    reduced_train_data,
    reduced_test_data,
    train_labels,
    test_labels
):
    """Train and evaluate a simple linear SVM classification pipeline.

    Before passing the data to the SVM module, this function scales the data such that the mean
    is 0 and the variance is 1.

    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        reduced_test_data (n_test x num_components matrix): reduced testing expression data
        train_labels (array of length n_train): array of cell type labels for training data
        test_labels (array of length n_test): array of cell type labels for testing data

    Return:
        (classifier, score): a tuple consisting of
            1. classifier: the trained classifier
            2. The score (accuracy) of the classifier on the test data.
    """
    classifier = make_pipeline(StandardScaler(), LinearSVC())
    classifier.fit(reduced_train_data, train_labels)
    print(classifier.score(reduced_train_data, train_labels))
    score = classifier.score(reduced_test_data, test_labels)
    print (classifier, score)
    return (classifier, score)


if __name__ == "__main__":
    train_gene_expression = np.load(sys.argv[1])["train"]
    test_gene_expression = np.load(sys.argv[2])["test"]
    train_labels = np.load(sys.argv[3]).astype(str)
    test_labels = np.load(sys.argv[4]).astype(str)

    top_gene_filter = get_top_gene_filter(train_gene_expression)
    filtered_test_gene_expression = test_gene_expression[:, top_gene_filter]
    filtered_train_gene_expression = train_gene_expression[:, top_gene_filter]

    mode = sys.argv[5]
    if mode == "svm_pipeline":
        reduced_train_data, reduced_test_data = reduce_dimensionality_pca(
            filtered_train_gene_expression,
            filtered_test_gene_expression,
            n_components=50
        )
        plot_transformed_cells(reduced_train_data, train_labels)
        train_and_evaluate_svm_classifier(
            reduced_train_data,
            reduced_test_data,
            train_labels,
            test_labels
        )