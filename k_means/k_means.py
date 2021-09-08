import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, K=2, max_iter=100):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.K = K
        self.max_iter = max_iter
        self.samples = None
        self.centroids = None
        self.centroid_assignments = None

    def assign_closest_centroid(self, samples):
        # assign closest centroid to every data point
        return np.array([np.argmin(np.square(np.linalg.norm(sample - self.centroids, axis=1))) for sample in samples])

    def init_centroids(self):
        # randomly initialise centroids
        idx = np.random.choice(self.samples.shape[0], self.K, replace=False)
        self.centroids = self.samples[idx]

    def optimise_centroids(self):
        # optimise centroids
        for k in range(self.K):
            self.centroids[k] = np.mean(self.samples[np.where(self.centroid_assignments == k)[0]], axis=0)


    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        samples = X.to_numpy()

        # data normalization
        samples -= samples.mean(axis=0)
        samples /= samples.std(axis=0)
        X = pd.DataFrame(samples, columns=['x0', 'x1'])
        self.samples = samples
        self.init_centroids()

        # TODO: iterate multiple times to find the optimal centroids
        current_iter = 0
        while current_iter < self.max_iter:
            self.centroid_assignments = self.assign_closest_centroid(self.samples)
            old_centroids = np.copy(self.centroids)
            self.optimise_centroids()
            C = self.centroids
            K = len(C)
            _, ax = plt.subplots(figsize=(5, 5), dpi=100)
            sns.scatterplot(x='x0', y='x1', hue=self.centroid_assignments, hue_order=range(K), palette='tab10', data=X, ax=ax)
            sns.scatterplot(x=C[:, 0], y=C[:, 1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black',
                            ax=ax)
            ax.legend().remove()
            plt.show()
            if np.all(old_centroids == self.centroids):
                break
            current_iter += 1



        return self.centroid_assignments

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """

        return self.assign_closest_centroid(X.to_numpy())

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points

    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
