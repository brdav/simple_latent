import numpy as np


class Bernoulli:
    """
    Bernoulli distribution.
    """

    def __init__(self, mu):
        """
        Construct Bernoulli distribution.

        Arguments
        ---------
        mu : (D,) np.ndarray
            probability of value 1 for each element
        """
        self.D = mu.shape[0]
        self.mu = mu

    def mle(self, X, weights=None):
        """
        Maximum likelihood estimate of the
        distribution parameters.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations
        weights : (N,) np.ndarray, optional
            sample weights
        """
        if weights is None:
            self.mu = np.mean(X, axis=0)
        else:
            self.mu = np.average(X, weights=weights, axis=0)

    def log_pdf(self, X):
        """
        Log probability density evaluated at input
        observations.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        log_prob : (N,) np.ndarray
            log probability density values
        """
        eps = np.finfo(float).eps
        mu_c = np.clip(self.mu, eps, 1 - eps)[None, :]
        return np.sum(X * np.log(mu_c) + (1 - X) * np.log(1 - mu_c), axis=1)

    def draw(self, N=1):
        """
        Sample from the distribution.

        Arguments
        ---------
        N : int
            sample size

        Returns
        -------
        sample : (N, D) np.ndarray
            sample from the Bernoulli distribution
        """
        return (self.mu > np.random.uniform(size=(N, self.D))).astype(int)

    def parameters(self):
        """
        Get parameters of the distribution.

        Returns
        -------
        params : (*,) np.ndarray
            parameters
        """
        return self.mu.ravel()
