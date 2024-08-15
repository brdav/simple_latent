import numpy as np


class LinearRegression:
    """Linear regression model.

    y = X @ w
    t ~ N(t|y, beta^-1)
    """

    def __init__(self, w, beta):
        """Initialize linear regression model.

        Arguments
        ---------
        w : (D,) np.ndarray
            model weights
        beta : float
            noise precision
        """
        self.D = w.shape[0]
        self.w = w
        self.beta = beta

    def mle(self, Xt, weights=None):
        """Perform least squares fitting.

        Arguments
        ---------
        Xt : ((N, D), (N,)) tuple of np.ndarray
            observations and targets
        weights : (N,) np.ndarray, optional
            sample weights
        """
        X, t = Xt
        if weights is None:
            self.w = np.linalg.pinv(X) @ t  # Eq. (3.15)
            self.beta = 1.0 / np.mean(np.square(X @ self.w - t))  # Eq. (3.21)
        else:
            # weighted least squares fitting
            self.w = np.linalg.lstsq(
                X.T @ np.diag(weights) @ X, X.T @ np.diag(weights) @ t
            )[0]
            self.beta = 1.0 / np.mean(weights * np.square(X @ self.w - t))

    def predict(self, X):
        """Return prediction given input.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        """
        return X @ self.w

    def log_pdf(self, Xt):
        """Log probability density / log predictive density for
        observations under the current model.

        Arguments
        ---------
        Xt : ((N, D), (N,)) tuple of np.ndarray
            observations and targets

        Returns
        -------
        log_prob : (N,) np.ndarray
            log probability density values
        """
        X, t = Xt
        maha_sq = np.square(t - X @ self.w) * self.beta
        return -0.5 * (maha_sq + np.log(2 * np.pi) - np.log(self.beta))

    def parameters(self):
        """
        Get parameters of the model.

        Returns
        -------
        params : (*,) np.ndarray
            parameters of the model
        """
        return np.hstack((self.w.ravel(), [self.beta]))
