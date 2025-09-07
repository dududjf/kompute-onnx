from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reduce_prod import ReduceProdOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_prod_op = ReduceProdOp(mgr)
x = np.random.random((255, 127, 63)).astype(np.float32)


def handle_axes(axes):
    if isinstance(axes, tuple):
        if len(axes) == 0:
            return None
        return axes
    if axes is None:
        return None
    if isinstance(axes, (int, tuple)):
        return axes
    if not isinstance(axes, np.ndarray):
        raise TypeError(f"axes must be an array, not {type(axes)}.")
    if len(axes.shape) == 0:
        return int(axes)
    if 0 in axes.shape:
        return None
    return tuple(axes.ravel().tolist())


def np_prod(data, axes=None, keepdims=1, noop_with_empty_axes=0):
    if axes is None and noop_with_empty_axes:
        return data
    keepdims = keepdims != 0
    axes = handle_axes(axes)
    try:
        res = np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype)
        if not keepdims and not isinstance(res, np.ndarray):
            res = np.array(res)
    except TypeError as e:
        raise TypeError(
            f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
        ) from e
    return res


# -------- Case 1: axis: None, keepdims: 1, noop_with_empty_axes: 0 --------
print("Case 1: axis: None, keepdims: 1, noop_with_empty_axes: 0")
start_time = time.time()
np_out = np_prod(x)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: axis: None, keepdims: 0, noop_with_empty_axes: 0 --------
print("Case 2: axis: None, keepdims: 0, noop_with_empty_axes: 0")
start_time = time.time()
np_out = np_prod(x, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x, None, 0)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: axes: None, keepdims: 1, noop_with_empty_axes: 1 --------
print("Case 3: axes: None, keepdims: 1, noop_with_empty_axes: 1")
start_time = time.time()
np_out = np_prod(x, None, keepdims=1, noop_with_empty_axes=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x, None, 1, 1)[0]
print(f"{reduce_prod_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: axes: None, keepdims: 0, noop_with_empty_axes: 1 --------
print("Case 4: axes: None, keepdims: 0, noop_with_empty_axes: 1")
start_time = time.time()
np_out = np_prod(x, None, keepdims=0, noop_with_empty_axes=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x, None, 0, 1)[0]
print(f"{reduce_prod_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: axes: [1,2], keepdims: 1, noop_with_empty_axes: 0 --------
print("Case 5: axes: [1,2], keepdims: 1, noop_with_empty_axes: 0")
axes = np.array([1, 2], dtype=np.int32)
start_time = time.time()
np_out = np_prod(x, axes)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x, axes)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6: axes: 1, keepdims: 0, noop_with_empty_axes: 0 --------
print("Case 6: axes: (1), keepdims: 0, noop_with_empty_axes: 0")
axes = (1,)
start_time = time.time()
np_out = np_prod(x, axes, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reduce_prod_op.run(x, axes, 0)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
