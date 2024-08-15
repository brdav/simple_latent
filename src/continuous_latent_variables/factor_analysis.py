import numpy as np

from ..components import Gaussian


class FactorAnalysis:

    def __init__(self, W, psi, mu):
        """
        Construct factor analysis model.
        p(z_n)         = N(z_n|0, I)
        p(x_n|z_n)     = N(x_n|W z_n + mu, diag(psi))

        Arguments
        ---------
        W : (D, M) np.ndarray
            emission matrix
        psi : (D,) np.ndarray
            variances of emission distribution
        mu : (D,) np.ndarray
            mean of data
        """
        self.M = W.shape[1]
        self.W = W
        self.psi = psi
        self.mu = mu

    def mle(self, X, iter_max=100):
        """
        Maximum likelihood estimation of factor analysis
        parameters with EM algorithm.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations
        iter_max : int
            maximum number of EM steps
        """
        for _ in range(iter_max):
            params = self.parameters()
            posterior_sufficient_stats = self.e_step(X)
            self.m_step(X, *posterior_sufficient_stats)
            if np.allclose(params, self.parameters()):
                break

    def e_step(self, X):
        """
        Estimate sufficient statistics of latent variable posterior.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        Ezn : (N, M) np.ndarray
            mean of latent variable posterior
        Eznzn : (N, M, M) np.ndarray
            second moment of latent variable posterior
        """
        Xc = X - self.mu
        G = np.linalg.inv(
            np.eye(self.M) + self.W.T @ np.diag(1 / self.psi) @ self.W
        )  # Eq. (12.68)
        Ezn = Xc @ np.diag(1 / self.psi) @ self.W @ G  # Eq. (12.66)
        Eznzn = G + Ezn[:, :, None] * Ezn[:, None, :]  # Eq. (12.67)
        return Ezn, Eznzn

    def m_step(self, X, Ezn, Eznzn):
        """
        Find parameters that maximize the expectation of the
        complete-data likelihood under the latent variable
        posterior.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations
        Ezn : (N, M) np.ndarray
            mean of latent variable posterior
        Eznzn : (N, M, M) np.ndarray
            second moment of latent variable posterior
        """
        # mu could be estimated just once
        self.mu = np.mean(X, axis=0)
        Xc = X - self.mu

        self.W = Xc.T @ Ezn @ np.linalg.inv(np.sum(Eznzn, axis=0))  # Eq. (12.69)
        S = np.cov(Xc, rowvar=False)
        self.psi = np.diag(S - self.W @ (Ezn.T @ Xc) / X.shape[0])  # Eq. (12.70)

    def log_likelihood(self, X):
        """
        Log likelihood of data under current model.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        log_l : float
            data log likelihood
        """
        C = self.W @ self.W.T + np.diag(self.psi)  # Eq. (12.65)
        return sum(Gaussian(self.mu, C).log_pdf(X))

    def parameters(self):
        """
        Get model parameters.

        Returns
        -------
        params : (*,) np.ndarray
            model parameters
        """
        return np.hstack(
            (
                self.W.ravel(),
                self.psi.ravel(),
                self.mu.ravel(),
            )
        )

    def draw(self, N=1):
        """
        Draw random data from this model through
        hierarchical sampling.

        Arguments
        ---------
        N : int
            sample size

        Returns
        -------
        X : (N, D) np.ndarray
            generated observations
        """
        X = []
        while len(X) < N:
            z_n = np.random.multivariate_normal(np.zeros(self.M), np.eye(self.M))
            X.append(
                np.random.multivariate_normal(self.W @ z_n + self.mu, np.diag(self.psi))
            )
        return np.stack(X, axis=0)
