import numpy as np


class KMeans:
    """
    K-means clustering.
    """

    def __init__(self, K):
        """
        Initialize K-means model.

        Parameters
        ----------
        K : int
            number of clusters
        """
        self.K = K
        self.mu = None

    def fit(self, X, iter_max=100):
        """
        Perform K-means algorithm.

        Parameters
        ----------
        X : (N, D) np.ndarray
            observations
        iter_max : int
            maximum number of iterations
        """
        self.mu = X[np.random.choice(len(X), self.K, replace=False)]
        for _ in range(iter_max):
            mu_prev = self.mu.copy()
            c_ind = np.argmin(
                np.sqrt(np.sum((X[:, None, :] - self.mu[None, :, :]) ** 2, axis=2)),
                axis=1,
            )
            self.mu = np.array([np.mean(X[c_ind == k], axis=0) for k in range(self.K)])
            if np.allclose(mu_prev, self.mu):
                break

    def transform(self, X):
        """
        Calculate closest cluster center index.

        Parameters
        ----------
        X : (N, D) np.ndarray
            input data

        Returns
        -------
        index : (N,) np.ndarray
            cluster membership of each observation
        """
        return np.argmin(
            np.sqrt(np.sum((X[:, None, :] - self.mu[None, :, :]) ** 2, axis=2)), axis=1
        )
