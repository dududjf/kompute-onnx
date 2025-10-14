import numpy as np
from kp import Manager
import time
from kp_onnx.kop_unique import UniqueOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

unique_op = UniqueOp(mgr)


def _specify_int64(indices, inverse_indices, counts):  # type: ignore
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


def np_unique(x, axis=None, sorted=None):
    if axis is None or np.isnan(axis):
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
    else:
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=axis)

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
    return y, indices, inverse_indices, counts


# x = np.random.random((32, 32, 8, 32)).astype(np.float32)

print("Case 1:")
x = [2, 1, 1, 3, 4, 3]

start_time = time.time()
np_out = np_unique(x, sorted=0)
print("Numpy:", time.time() - start_time, "seconds")

print("np_out:", np_out)

start_time = time.time()
unique_op.axis = None
unique_op.sorted = 0
kp_out = unique_op.run(x)[0]
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("Max error:", np.abs(np_out[i] - kp_out[i]).max())
    print(np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')