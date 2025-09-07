from kp import Manager
import numpy as np
import time
from kp_onnx.kop_argmax import ArgMaxOp


def numpy_argmax(x: np.ndarray, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> np.ndarray:
    if select_last_index == 0:
        result = np.argmax(x, axis=axis)
        if keepdims and len(result.shape) < len(x.shape):
            result = np.expand_dims(result, axis=axis)
        return result.astype(np.int64)
    else:
        data = np.flip(x, axis)
        result = np.argmax(data, axis=axis)
        result = data.shape[axis] - result - 1
        if keepdims:
            result = np.expand_dims(result, axis=axis)
        return result.astype(np.int64)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

argmax_op = ArgMaxOp(mgr)

print("Case 1: 1D Tensor (1024,), axis: None, keepdims: None, select_last_index: None")
x = np.random.uniform(-5.0, 5.0, (1024,)).astype(np.float32)

start_time = time.time()
np_out = numpy_argmax(x)
print(f"Numpy time: {time.time() - start_time} seconds")

start_time = time.time()
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print("Case 2: 3D Tensor (32, 64, 128), axis: 1, keepdims: None, select_last_index: None")
x = np.random.uniform(-5.0, 5.0, (32, 64, 128)).astype(np.float32)
axis = 1

start_time = time.time()
np_out = numpy_argmax(x, axis=axis)
print(f"Numpy time: {time.time() - start_time} seconds")

start_time = time.time()
kp_out = argmax_op.run(x, axis)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print("Case 3: 3D Tensor (32, 64, 128), axis: -1, keepdims: 0, select_last_index: None")
x = np.random.uniform(-3.0, 3.0, (1024,)).astype(np.float32)
axis = -1
keepdims = 0

# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=axis, keepdims=keepdims)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
kp_out = argmax_op.run(x, axis, keepdims)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print("Case 4: 3D Tensor (32, 64, 128), axis: 2, keepdims: 0, select_last_index: 1")
x = np.random.uniform(-3.0, 3.0, (32, 64, 128)).astype(np.float32)
x[:, :, 127] = 10  # 在 axis=2 的最后位置设最大值
x[:, :, 126] = 10  # 在 axis=2 的倒数第二位置设最大值
axis = 2
keepdims = 0
select_last_index = 1

# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
kp_out = argmax_op.run(x, axis, keepdims, select_last_index)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))