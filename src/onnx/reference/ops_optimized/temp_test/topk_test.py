import numpy as np


def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    """See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/blob/main/
    sklearn/neighbors/_base.py#L304>`_.
    """
    if isinstance(k, np.ndarray):
        if k.size != 1:
            raise RuntimeError(f"k must be an integer not {k!r}.")
        k = k[0]
    # This conversion is needed for distribution x86.
    k = int(k)
    # Used to tiebreak
    ind_axis = np.indices(X.shape)[axis]
    if largest:
        ind_axis = -ind_axis
    sorted_indices = np.lexsort((ind_axis, X), axis=axis)
    sorted_values = np.take_along_axis(X, sorted_indices, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    ark = np.arange(k)
    topk_sorted_indices = np.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = np.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices

'''
沿指定轴检索前 K 个最大或最小的元素。
如果“largest”为 1（默认值），则返回 k 个最大的元素。
如果“sorted”为 1（默认值），则返回的 k 个元素将进行排序。

属性：axis(默认: -1), largest(默认: 1), sorted(默认: 1)
输入：x, k
'''

X = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)
print(topk_sorted_implementation(X, 2, 0, True))
print('----')

X = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)
print(topk_sorted_implementation(X, 0, 1, True))