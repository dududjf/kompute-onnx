import numpy as np


def _rms_normalization(X: np.ndarray, W: np.ndarray, axis: int = -1, epsilon: float = 1e-5) -> np.ndarray:
    shape = X.shape
    rank = len(shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + rank

    # This computes RMS for every x_mat's column.
    x_squared = np.power(X, 2)
    x_squared_mean = np.mean(
        x_squared, axis=tuple(range(axis, len(shape))), keepdims=True
    )
    # epsilon adjustment to avoid divide-by-zero.
    rmseps = x_squared_mean + epsilon
    rms = np.sqrt(rmseps)
    rms_reciprocal = np.reciprocal(rms)

    y_mat = X * rms_reciprocal
    # W is linear coefficient.
    Y = y_mat * W

    return Y.astype(X.dtype)

def _run(X, Scale, axis=None, epsilon=None, stash_type=None):
    if stash_type != 1:
        raise NotImplementedError(
            f"RMSNormalization not implemented for stash_type={stash_type} != 1."
        )
    res = _rms_normalization(X, Scale, axis=axis, epsilon=epsilon)
    return res
'''
属性: axis(默认: -1), epsilon(默认: 1e-5), stash_type(默认: 1)
输入: x, scale
'''

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
print(_run(x, 5, axis=0, epsilon=1e-5, stash_type=1))