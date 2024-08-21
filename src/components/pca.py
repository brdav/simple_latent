import numpy as np


class PCA:
    """
    Principle component analysis.
    """

    def __init__(self, M):
        """
        Construct principal component analysis.

        Arguments
        ---------
        M : int
            number of components
        """
        self.M = M

    def fit(self, X):
        """
        Fit PCA.

        Parameters
        ----------
        X : (N, D) np.ndarray
            observations
        """
        self.mu = np.mean(X, axis=0)
        Xc = X - self.mu
        _, _, VT = np.linalg.svd(Xc, full_matrices=False)
        self.W = VT[: self.M].T

    def transform(self, X):
        """
        Project data into latent space.

        Parameters
        ----------
        X : (N, D) np.ndarray
            observations
        """
        return (X - self.mu) @ self.W
