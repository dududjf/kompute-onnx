import numpy as np


def _specify_int64(indices, inverse_indices, counts):  # type: ignore
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


def _run(x, axis=None, sorted=None):  # type: ignore  # noqa: A002
    if axis is None or np.isnan(axis):
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
    else:
        y, indices, inverse_indices, counts = np.unique(
            x, True, True, True, axis=axis
        )
    if len(self.onnx_node.output) == 1:
        return (y,)

    if not sorted:
        argsorted_indices = np.argsort(indices)
        inverse_indices_map = dict(
            zip(argsorted_indices, np.arange(len(argsorted_indices)))
        )
        indices = indices[argsorted_indices]
        y = np.take(x, indices, axis=0)
        inverse_indices = np.asarray(
            [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64
        )
        counts = counts[argsorted_indices]

    indices, inverse_indices, counts = _specify_int64(
        indices, inverse_indices, counts
    )
    # numpy 2.0 has a different behavior than numpy 1.x.
    inverse_indices = inverse_indices.reshape(-1)
    if len(self.onnx_node.output) == 2:
        return (y, indices)
    if len(self.onnx_node.output) == 3:
        return (y, indices, inverse_indices)
    return (y, indices, inverse_indices, counts)


'''
查找张量的唯一元素，或唯一子张量

属性: axis(可选), sorted(可选, 默认: 1)
输入: x
输出: y, indices, inverse_indices, counts
'''


x = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)
print(_run(x, axis=None, sorted=None))