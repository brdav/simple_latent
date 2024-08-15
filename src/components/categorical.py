import numpy as np


class Categorical:
    """
    Categorical/multinoulli distribution.
    """

    def __init__(self, mu):
        """
        Construct categorical distribution.

        Arguments
        ---------
        mu : (D,) np.ndarray
            probability of each category
        """
        self.D = mu.shape[0]
        self.mu = mu

    def mle(self, X, weights=None):
        """
        Maximum likelihood estimate of the
        distribution parameters.

        Arguments
        ---------
        X : (N,) np.ndarray
            observations
        weights : (N,) np.ndarray, optional
            sample weights
        """
        X_onehot = np.eye(self.D)[X]
        if weights is None:
            self.mu = np.mean(X_onehot, axis=0)
        else:
            self.mu = np.average(X_onehot, weights=weights, axis=0)

    def log_pdf(self, X):
        """
        Log probability density evaluated at input
        observations.

        Arguments
        ---------
        X : (N,) np.ndarray
            observations

        Returns
        -------
        log_prob : (N,) np.ndarray
            log probability density values
        """
        eps = np.finfo(float).eps
        mu_c = np.clip(self.mu, eps, None)
        return np.log(mu_c[X])

    def draw(self, N=1):
        """
        Sample from the distribution.

        Arguments
        ---------
        N : int
            sample size

        Returns
        -------
        sample : (N,) np.ndarray
            sample from the categorical distribution
        """
        return np.random.choice(self.mu.shape[0], N, p=self.mu)

    def parameters(self):
        """
        Get parameters of the distribution.

        Returns
        -------
        params : (*,) np.ndarray
            parameters
        """
        return self.mu.ravel()