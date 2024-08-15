import numpy as np

from ..components import Gaussian, logsumexp


class GaussianMixtureModel:

    def __init__(self, pi, mu, Sigma):
        """
        Construct Gaussian mixture model.
        p(z_n)         = Categorical(pi)
        p(x_n|z_n)     = N(x_n|mu_{z_n}, Sigma_{z_n})

        Arguments
        ---------
        pi : (K,) np.ndarray
            mixing coefficients
        mu : (K, D) np.ndarray
            means of emission models
        Sigma : (K, D, D) np.ndarray
            covariance matrices of emission models
        """
        self.K = pi.shape[0]
        self.pi = pi
        self.emission_models = [Gaussian(m, s) for m, s in zip(mu, Sigma)]

    def mle(self, X, iter_max=100):
        """
        Maximum likelihood estimation of model parameters
        with EM algorithm.

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
        Ezn : (N, K) np.ndarray
            posterior mean of latent variables
        """
        log_gamma = np.stack(
            [np.log(p) + c.log_pdf(X) for p, c in zip(self.pi, self.emission_models)],
            axis=1,
        )
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
        return (np.exp(log_gamma),)

    def m_step(self, X, Ezn):
        """
        Find parameters that maximize the expectation of the
        complete-data likelihood under the latent variable
        posterior.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations
        Ezn : (N, K) np.ndarray
            posterior mean of latent variables
        """
        self.pi = np.mean(Ezn, axis=0)  # Eq. (9.26)
        for i, c in enumerate(self.emission_models):
            # maximize weighted log likelihood for emission models
            c.mle(X, weights=Ezn[:, i])

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
        return sum(
            logsumexp(
                [
                    np.log(p) + c.log_pdf(X)
                    for p, c in zip(self.pi, self.emission_models)
                ],
                axis=0,
            )
        )

    def parameters(self):
        """
        Get parameters of the mixture model.

        Returns
        -------
        params : (*,) np.ndarray
            parameters
        """
        return np.concatenate(
            [np.ravel([c.parameters() for c in self.emission_models]), self.pi]
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
            z_n = np.random.choice(self.K, p=self.pi)
            X.append(self.emission_models[z_n].draw())
        return np.concatenate(X)
