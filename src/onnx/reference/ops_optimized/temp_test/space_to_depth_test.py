import numpy as np


def _run(data, blocksize=None):  # type: ignore
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, C, H, W = data.shape
    tmpshape = (
        b,
        C,
        H // blocksize,
        blocksize,
        W // blocksize,
        blocksize,
    )
    reshaped = np.reshape(data, tmpshape)
    transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        C * blocksize * blocksize,
        H // blocksize,
        W // blocksize,
    )
    y = np.reshape(transposed, finalshape).astype(data.dtype)
    return y


x = np.random.random_integers(-2, 2, (1, 1, 3, 3)).astype(np.float32)
print(x)
print('----')
print(_run(x, blocksize=3))