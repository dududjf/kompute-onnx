import numpy as np


def _slice(data: np.ndarray, starts: np.ndarray, ends: np.ndarray, axes: np.ndarray, steps: np.ndarray) -> np.ndarray:
    if isinstance(starts, list):
        starts = np.array(starts)
    if isinstance(ends, list):
        ends = np.array(ends)
    if isinstance(axes, list):
        axes = np.array(axes)
    if isinstance(steps, list):
        steps = np.array(steps)
    if len(starts.shape) == 0:
        starts = np.array([starts])
    if len(ends.shape) == 0:
        ends = np.array([ends])
    if axes is None:
        if steps is None:
            slices = [slice(s, e) for s, e in zip(starts, ends)]
        else:
            slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
    else:  # noqa: PLR5501
        if steps is None:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a in zip(starts, ends, axes):
                slices[a] = slice(s, e)
        else:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a, d in zip(starts, ends, axes, steps):
                slices[a] = slice(s, e, d)
    try:
        return data[tuple(slices)]  # type: ignore
    except TypeError as e:  # pragma: no cover
        raise TypeError(
            f"Unable to extract slice {slices!r} for shape {data.shape!r}."
        ) from e


x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print('----')
print(_slice(x, starts=[2, 1], ends=[1, 2], axes=[0, 1], steps=[-1, 1]))