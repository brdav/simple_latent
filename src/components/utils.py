import numpy as np


def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=axis)) + np.squeeze(a_max, axis=axis)


def sigmoid(a):
    return np.tanh(a * 0.5) * 0.5 + 0.5
