import numpy as np


def _run(data, blocksize=None, mode=None):  # type: ignore
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, c, h, w = data.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        # assert mode == "CRD"
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    y = np.reshape(transposed, finalshape)
    return y

'''
将数据从深度维度重新排列（置换）到空间数据块中。这是 SpaceToDepth 的逆变换。
DCR 模式下，输入张量沿深度维度的元素按以下顺序重新排列：深度、列
在 CRD 模式下，输入张量沿深度维度的元素按以下顺序重新排列：列、行，然后是深度
'''

x = np.random.random_integers(-2, 2, (1, 12, 4, 4)).astype(np.float32)

print(_run(x, blocksize=2, mode="DCR"))