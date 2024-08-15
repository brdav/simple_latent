import numpy as np

from ..components import Gaussian


class GaussianHiddenMarkovModel:

    def __init__(self, pi, A, mu, Sigma):
        """
        Construct hidden Markov model with Gaussian emission model.
        p(z_1)         = Categorical(pi)
        p(z_n|z_{n-1}) = Categorical(A z_{n-1})
        p(x_n|z_n)     = N(x_n|mu_{z_n}, Sigma_{z_n})

        Arguments
        ---------
        pi : (K,) np.ndarray
            initial probability of each hidden state
        A : (K, K) np.ndarray
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        mu : (K, D) np.ndarray
            means of emission models
        Sigma : (K, D, D) np.ndarray
            covariance matrices of emission models
        """
        self.K = pi.shape[0]
        self.pi = pi
        self.A = A
        self.emission_models = [Gaussian(m, s) for m, s in zip(mu, Sigma)]

    def mle(self, X, iter_max=100):
        """
        Maximum likelihood estimation of HMM parameters with EM algorithm.

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
        Estimate sufficient statistics of latent variable posterior
        with forward-backward algorithm (a.k.a. sum-product
        algorithm).

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        Ezn : (N, K) np.ndarray
            posterior mean of latent variables
        Eznzn_1 : (N - 1, K, K) np.ndarray
            posterior transition probability between adjacent latent variables
        """
        alpha_hat, emission_p, c = self.forward(X)
        beta_hat = self.backward(emission_p, c)

        Ezn = alpha_hat * beta_hat  # Eq. (13.64)
        Eznzn_1 = (
            self.A
            * emission_p[1:, None, :]
            * beta_hat[1:, None, :]
            * alpha_hat[:-1, :, None]
        ) / c[
            1:, None, None
        ]  # Eq. (13.65)
        return Ezn, Eznzn_1

    def m_step(self, X, Ezn, Eznzn_1):
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
        Eznzn_1 : (N - 1, K, K) np.ndarray
            posterior transition probability between adjacent latent variables
        """
        self.pi = Ezn[0]  # Eq. (13.18)
        self.A = np.sum(Eznzn_1, axis=0) / np.sum(Eznzn_1, axis=(0, 2))  # Eq. (13.19)
        for i, c in enumerate(self.emission_models):
            # maximize weighted log likelihood for emission models
            c.mle(X, weights=Ezn[:, i])

    def forward(self, X):
        """
        Forward recursion of sum-product algorithm.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        alpha_hat : (N, K) np.ndarray
            normalized local posterior marginals
        emission_p : (N, K) np.ndarray
            emission probabilities
        c : (N,) np.ndarray
            normalization constants of alpha_hat
        """
        emission_p = np.exp(
            np.stack(
                [m.log_pdf(X) for m in self.emission_models],
                axis=1,
            )
        )
        alpha = self.pi * emission_p[0]  # Eq. (13.37)
        # introduce scaling factors according to Sec. 13.2.4:
        c = [alpha.sum()]
        alpha_hat = [alpha / alpha.sum()]
        for emission_lh_i in emission_p[1:]:
            alpha = alpha_hat[-1] @ self.A * emission_lh_i  # Eq. (13.59)
            c.append(alpha.sum())
            alpha_hat.append(alpha / alpha.sum())
        return np.asarray(alpha_hat), emission_p, np.asarray(c)

    def backward(self, emission_p, c):
        """
        Backward recursion of sum-product algorithm.

        Arguments
        ---------
        emission_p : (N, K) np.ndarray
            emission probabilities
        c : (N,) np.ndarray
            normalization constants of alpha_hat

        Returns
        -------
        beta_hat : (N, K) np.ndarray
            normalized local posterior marginals
        """
        beta_hat = [np.ones(self.K)]
        for emission_lh_i, c_i in zip(emission_p[-1:0:-1], c[-1:0:-1]):
            beta_hat.insert(
                0, self.A @ (emission_lh_i * beta_hat[0]) / c_i
            )  # Eq. (13.62)
        return np.asarray(beta_hat)

    def log_likelihood(self, X):
        """
        Log likelihood of sequence under current HMM.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        log_l : float
            data log likelihood
        """
        _, _, c = self.forward(X)
        return sum(np.log(c))  # Eq. (13.63)

    def parameters(self):
        """
        Get parameters of the HMM.

        Returns
        -------
        params : (*,) np.ndarray
            parameters
        """
        return np.concatenate(
            [np.ravel([c.parameters() for c in self.emission_models]), self.pi]
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
        z_n = np.random.choice(self.K, p=self.pi)
        X = []
        while len(X) < N:
            X.append(self.emission_models[z_n].draw())
            z_n = np.random.choice(self.K, p=self.A[z_n])
        return np.concatenate(X)

    def viterbi(self, X):
        """
        Viterbi algorithm (a.k.a. max-sum algorithm).

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        z_hid : (N,) np.ndarray
            the most probable sequence of hidden variables
        """
        log_emission_p = np.stack(
            [m.log_pdf(X) for m in self.emission_models],
            axis=1,
        )
        omega = np.log(self.pi) * log_emission_p[0]  # Eq. (13.69)
        phi = []
        for i in range(1, len(X)):
            omega_tmp = omega[:, None] + np.log(self.A) + log_emission_p[i]
            omega = np.max(omega_tmp, axis=0)  # Eq. (13.68)
            index = np.argmax(omega_tmp, axis=0)
            phi.append(index)
        z_hid = [np.argmax(omega)]
        for phi_i in phi[::-1]:
            z_hid.insert(0, phi_i[z_hid[0]])
        return z_hid
