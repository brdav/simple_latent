import numpy as np


class Gaussian:
    """
    Gaussian distribution.
    """

    def __init__(self, mu, Sigma):
        """
        Initialize a Gaussian distribution.

        Arguments
        ----------
        mu : (D,) np.ndarray
            mean of Gaussian
        Sigma : (D, D) np.ndarray
            covariance matrix of Gaussian
        """
        self.D = mu.shape[0]
        self.mu = mu
        self.Sigma = Sigma

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
            diff = X - self.mu
            self.Sigma = np.mean(diff[:, :, None] * diff[:, None, :], axis=0)
        else:
            self.mu = np.average(X, weights=weights, axis=0)
            diff = X - self.mu
            self.Sigma = np.average(
                diff[:, :, None] * diff[:, None, :], weights=weights, axis=0
            )

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
        cov_chol = np.linalg.cholesky(self.Sigma)
        cov_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))
        cov_sol = np.linalg.solve(cov_chol, (X - self.mu).T).T
        return -0.5 * (
            np.sum(cov_sol**2, axis=1) + self.D * np.log(2 * np.pi) + cov_log_det
        )

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
            sample from the Gaussian distribution
        """
        return np.random.multivariate_normal(self.mu, self.Sigma, N)

    def parameters(self):
        """
        Get parameters of the distribution.

        Returns
        -------
        params : (*,) np.ndarray
            parameters
        """
        return np.hstack((self.mu.ravel(), self.Sigma.ravel()))
