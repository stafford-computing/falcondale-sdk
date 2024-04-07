"""
Auxiliary functions
"""

import numpy as np
from pandas import DataFrame


def prob(dataset: DataFrame, max_bins=10):
    """Joint probability distribution P(X) for the given data.

    Parameters:
    dataset (DataFrame): Input DataFrame
    max_bins (int): Maximal number of bins to discretize the sample

    Returns:
    list: Join probability distribution
    """

    # bin by the number of different values per feature
    _, num_columns = dataset.shape
    bins = [min(len(np.unique(dataset[:, ci])), max_bins) for ci in range(num_columns)]

    freq, _ = np.histogramdd(dataset, bins)
    joint_prob = freq / np.sum(freq)
    return joint_prob


def shannon_entropy(joint_prob) -> int:
    """Shannon entropy H(X) is the sum of P(X)log(P(X)) for probabilty distribution P(X)."""
    flatten_probs = joint_prob.flatten()
    return -sum(pi * np.log2(pi) for pi in flatten_probs if pi)


def conditional_shannon_entropy(p, *conditional_indices):
    """Shannon entropy of P(X) conditional on variable j"""

    axis = tuple(i for i in np.arange(len(p.shape)) if i not in conditional_indices)

    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))


def mutual_information(p, j):
    """Mutual information between all variables and variable j"""
    return shannon_entropy(np.sum(p, axis=j)) - conditional_shannon_entropy(p, j)


def conditional_mutual_information(p, j, *conditional_indices):
    """Mutual information between variables X and variable Y conditional on variable Z."""

    marginal_conditional_indices = [i - 1 if i > j else i for i in conditional_indices]

    return conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices) - conditional_shannon_entropy(
        p, j, *conditional_indices
    )
