from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reducemean import ReduceMeanOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_mean_op = ReduceMeanOp(mgr)


def np_mean(data, axis=None, keepdims=1, noop_with_empty_axes=0):
    if axis is None and noop_with_empty_axes:
        return data
    keepdims = keepdims != 0
    try:
        output = np.mean(data, axis=axis, keepdims=keepdims, dtype=data.dtype)
    except TypeError as e:
        raise TypeError(
            f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
        ) from e
    return output

# -------- Case 1: axis: None, keepdims: 1, noop_with_empty_axes: 0 --------
print("Case 1: axis: None, keepdims: 1, noop_with_empty_axes: 0")
x = np.random.random((3, 3, 3)).astype(np.float32)

start_time = time.time()
np_out = np_mean(x)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("x:", x)
print("np_out:", np_out)
print("kp_out:", kp_out)

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: axis: None, keepdims: 0, noop_with_empty_axes: 0 --------
print("Case 2: axis: None, keepdims: 0, noop_with_empty_axes: 0")
x = np.random.random((3, 1024, 1024)).astype(np.float32)

start_time = time.time()
np_out = np_mean(x, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x, 0)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: axes: None, keepdims: 1, noop_with_empty_axes: 1 --------
print("Case 3: axes: None, keepdims: 1, noop_with_empty_axes: 1")
x = np.random.random((3, 1024, 1024)).astype(np.float32)

start_time = time.time()
np_out = np_mean(x, None, keepdims=1, noop_with_empty_axes=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x, None, 1, 1)[0]
print(f"{reduce_mean_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: axes: None, keepdims: 0, noop_with_empty_axes: 1 --------
print("Case 4: axes: None, keepdims: 0, noop_with_empty_axes: 1")
x = np.random.random((3, 1024, 1024)).astype(np.float32)

start_time = time.time()
np_out = np_mean(x, None, keepdims=0, noop_with_empty_axes=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x, None, 0, 1)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: axes: [1,2], keepdims: 1, noop_with_empty_axes: 0 --------
print("Case 5: axes: [1,2], keepdims: 1, noop_with_empty_axes: 0")
x = np.random.random((3, 1024, 1024)).astype(np.float32)
axes = np.array([1, 2], dtype=np.float32)
start_time = time.time()
np_out = np_mean(x, axes)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x, axes)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# -------- Case 6: axes: [1,2], keepdims: 0, noop_with_empty_axes: 0 --------
print("Case 6: axes: [1,2], keepdims: 0, noop_with_empty_axes: 0")
x = np.random.random((3, 1024, 1024))
axes = np.array([1, 2], dtype=np.float32)
start_time = time.time()
np_out = np_mean(x, axes, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_mean_op.run(x, axes, 0)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

