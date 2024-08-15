import numpy as np

from ..components import Gaussian


class LinearDynamicalSystem:

    def __init__(self, C, Sigma, A, Gamma, mu0, P0):
        """
        Construct linear dynamical system (linear-Gaussian
        state space model).
        p(z_1)         = N(z_1|mu_0, P_0)
        p(z_n|z_{n-1}) = N(z_n|A z_{n-1}, Gamma)
        p(x_n|z_n)     = N(x_n|C z_n, Sigma)

        Arguments
        ---------
        C : (D, M) np.ndarray
            emission matrix
        Sigma : (D, D) np.ndarray
            covariance matrix of emission distribution
        A : (M, M) np.ndarray
            transition matrix
        Gamma : (M, M) np.ndarray
            covariance matrix of transition distribution
        mu0 : (M,) np.ndarray
            mean of initial latent variable
        P0 : (M, M) np.ndarray
            covariance matrix of initial latent variable
        """
        self.M = C.shape[1]
        self.C = C
        self.Sigma = Sigma
        self.A = A
        self.Gamma = Gamma
        self.mu0 = mu0
        self.P0 = P0

    def mle(self, X, iter_max=100):
        """
        Maximum likelihood estimation of LDS parameters
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
        Estimate sufficient statistics of latent variable
        posterior with forward-backward algorithm (a.k.a.
        sum-product algorithm, a.k.a. Kalman smoothing).

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        Ezn : (N, M) np.ndarray
            mean of latent variable posterior
        Eznzn_1 : (N - 1, M, M) np.ndarray
            pairwise posterior marginal of adjacent latent variables
        Eznzn : (N, M, M) np.ndarray
            second moment of latent variable posterior
        """
        mu, V, P, _ = self.forward(X)
        mu_hat, V_hat, J = self.backward(mu, V, P)
        Ezn = mu_hat  # Eq. (13.105)
        Eznzn_1 = (
            np.einsum("nij,nkj->nik", V_hat[1:], J[:-1])
            + Ezn[1:, :, None] * Ezn[:-1, None, :]
        )  # Eq. (13.106)
        Eznzn = V_hat + Ezn[:, :, None] * Ezn[:, None, :]  # Eq. (13.107)
        return Ezn, Eznzn_1, Eznzn

    def m_step(self, X, Ezn, Eznzn_1, Eznzn):
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
        Eznzn_1 : (N - 1, M, M) np.ndarray
            pairwise posterior marginal of adjacent latent variables
        Eznzn : (N, M, M) np.ndarray
            second moment of latent variable posterior
        """
        self.C = X.T @ Ezn @ np.linalg.pinv(np.sum(Eznzn, axis=0))  # Eq. (13.115)
        self.Sigma = np.mean(
            np.einsum("ni,nj->nij", X, X)
            - np.einsum("ij,nj,nk->nik", self.C, Ezn, X)
            - np.einsum("ni,nj,kj->nik", X, Ezn, self.C)
            + np.einsum("ij,njk,lk->nil", self.C, Eznzn, self.C),
            axis=0,
        )  # Eq. (13.116)
        self.Sigma = (self.Sigma + self.Sigma.T) / 2  # improves numerical stability

        self.A = np.sum(Eznzn_1, axis=0) @ np.linalg.pinv(
            np.sum(Eznzn[:-1], axis=0)
        )  # Eq. (13.113)
        self.Gamma = np.mean(
            Eznzn[1:]
            - np.einsum("ij,nkj->nik", self.A, Eznzn_1)
            - np.einsum("nij,kj->nik", Eznzn_1, self.A)
            + np.einsum("ij,njk,lk->nil", self.A, Eznzn[:-1], self.A),
            axis=0,
        )  # Eq. (13.114)
        self.Gamma = (self.Gamma + self.Gamma.T) / 2  # improves numerical stability

        self.mu0 = Ezn[0]  # Eq. (13.110)
        self.P0 = Eznzn[0] - Ezn[0, :, None] * Ezn[0, None, :]  # Eq. (13.111)

    def forward(self, X):
        """
        Perform forward recursion of sum-product algorithm
        (a.k.a. Kalman filtering).

        Arguments
        ---------
        X : (N, D) np.ndarray
            sequence of observations

        Returns
        -------
        mu : (N, M) np.ndarray
            filtering mean of latent variable sequence
        V : (N, M, M) np.ndarray
            filtering covariance of latent variable sequence
        P : (N, M, M) np.ndarray
            predicted covariance of latent variable sequence
        log_c : (N,) np.ndarray
            log normalization coefficients
        """
        K = (
            self.P0
            @ self.C.T
            @ np.linalg.pinv(self.C @ self.P0 @ self.C.T + self.Sigma)
        )  # Eq. (13.97)
        mu = [self.mu0 + K @ (X[0] - self.C @ self.mu0)]  # Eq. (13.94)
        V = [self.P0 - K @ self.C @ self.P0]  # Eq. (13.95)
        P = [self.A @ V[0] @ self.A.T + self.Gamma]  # Eq. (13.88)
        log_c = [
            Gaussian(self.C @ self.mu0, self.C @ self.P0 @ self.C.T + self.Sigma)
            .log_pdf(X[0][None, :])
            .item()
        ]

        for x_n in X[1:]:
            K = (
                P[-1]
                @ self.C.T
                @ np.linalg.pinv(self.C @ P[-1] @ self.C.T + self.Sigma)
            )  # Eq. (13.92)
            mu_n = self.A @ mu[-1] + K @ (x_n - self.C @ self.A @ mu[-1])  # Eq. (13.89)
            V_n = P[-1] - K @ self.C @ P[-1]
            P_n = self.A @ V_n @ self.A.T + self.Gamma  # Eq. (13.88)
            mu.append(mu_n)
            V.append(V_n)
            P.append(P_n)
            log_c.append(
                Gaussian(
                    self.C @ self.A @ mu[-1], self.C @ P[-1] @ self.C.T + self.Sigma
                )
                .log_pdf(x_n[None, :])
                .item()
            )
        return np.asarray(mu), np.asarray(V), np.asarray(P), np.asarray(log_c)

    def backward(self, mu, V, P):
        """
        Perform backward recursion of sum-product algorithm
        (a.k.a. Kalman smoothing when combined with forward recursion).

        Arguments
        ---------
        mu : (N, M) np.ndarray
            forward means of hidden variables
        V : (N, M, M) np.ndarray
            forward covariances of hidden variables
        P : (N, M, M) np.ndarray
            forward predicted covariances of hidden variables

        Returns
        -------
        mu_hat : (N, M) np.ndarray
            smoothing mean of latent variable sequence
        V_hat : (N, M, M) np.ndarray
            smoothing covariance of latent variable sequence
        J : (N, M, M) np.ndarray
            smoothing gain of latent variable sequence
        """
        mu_hat = [mu[-1]]
        V_hat = [V[-1]]
        J = [V[-1] @ self.A.T @ np.linalg.pinv(P[-1])]
        index = -2
        while index >= -len(mu):
            J_n = V[index] @ self.A.T @ np.linalg.pinv(P[index])  # Eq. (13.102)
            mu_hat_n = mu[index] + J_n @ (
                mu_hat[0] - self.A @ mu[index]
            )  # Eq. (13.100)
            V_hat_n = V[index] + J_n @ (V_hat[0] - P[index]) @ J_n.T  # Eq. (13.101)
            mu_hat.insert(0, mu_hat_n)
            V_hat.insert(0, V_hat_n)
            J.insert(0, J_n)
            index -= 1
        return np.asarray(mu_hat), np.asarray(V_hat), np.asarray(J)

    def log_likelihood(self, X):
        """
        Log likelihood of data under current LDS.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        log_l : float
            data log likelihood
        """
        _, _, _, log_c = self.forward(X)
        return sum(log_c)  # Eq. (13.63)

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
                self.C.ravel(),
                self.Sigma.ravel(),
                self.A.ravel(),
                self.Gamma.ravel(),
                self.mu0.ravel(),
                self.P0.ravel(),
            )
        )

    def draw(self, N=100):
        """
        Draw random sequence from this model through
        hierarchical sampling.

        Arguments
        ---------
        N : int
            length of the random sequence

        Returns
        -------
        X : (N, D) np.ndarray
            generated random sequence
        """
        z_n = np.random.multivariate_normal(self.mu0, self.P0)
        X = []
        while len(X) < N:
            X.append(np.random.multivariate_normal(self.C @ z_n, self.Sigma))
            z_n = np.random.multivariate_normal(self.A @ z_n, self.Gamma)
        return np.stack(X, axis=0)
