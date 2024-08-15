import numpy as np

from .utils import sigmoid


class LogisticRegression:
    """Logistic regression model.

    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def __init__(self, w):
        """Initialize logistic regression model.

        Arguments
        ---------
        w : (D,) np.ndarray
            model weights
        """
        self.D = w.shape[0]
        self.w = w

    def mle(self, Xt, weights=None, iter_max=100):
        """Maximum likelihood estimation of logistic regression model.

        Arguments
        ---------
        Xt : ((N, D), (N,)) tuple of np.ndarray
            observations and targets
        weights : (N,) np.ndarray, optional
            sample weights
        iter_max : int, optional
            maximum number of parameter update iteration
        """
        X, t = Xt
        for _ in range(iter_max):
            w_prev = np.copy(self.w)
            y = sigmoid(X @ self.w)
            if weights is None:
                grad = X.T @ (y - t)  # Eq. (4.96)
                H = (X.T * y * (1 - y)) @ X  # Eq. (4.97)
            else:
                grad = X.T @ (weights * (y - t))  # Eq. (14.51)
                H = (X.T * weights * y * (1 - y)) @ X  # Eq. (14.52)
            self.w -= np.linalg.lstsq(H, grad)[0]
            if np.allclose(self.w, w_prev):
                break

    def log_pdf(self, Xt):
        """Log probability density / log predictive density for
        observations under the current model.

        Arguments
        ---------
        Xt : ((N, D), (N,)) tuple of np.ndarray
            observations and targets

        Returns
        -------
        (N,) np.ndarray
            log probability density for each observation
        """
        X, t = Xt
        logits = X @ self.w
        return np.where(
            t == 1,
            (logits - np.log1p(np.exp(logits))),
            (-logits - np.log1p(np.exp(-logits))),
        )

    def predict(self, X):
        """Return probability of input belonging class 1.

        Arguments
        ---------
        X : (N, D) np.ndarray
            observations

        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        return sigmoid(X @ self.w)

    def parameters(self):
        """
        Get parameters of the model.

        Returns
        -------
        (*,) np.ndarray
            parameters of the model
        """
        return self.w.ravel()
