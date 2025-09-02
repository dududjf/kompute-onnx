from kp import Manager
import numpy as np
import time
from kp_onnx.kop_argmin import ArgMinOp

def onnx_argmin(data, axis=0, keepdims=True):
    result = np.argmin(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

argmin_op = ArgMinOp(mgr)

# Case 1: 1D 张量，轴 0
print("Case 1: 1D 张量，轴 0")
numpy_in = np.random.uniform(-8, 8, (1024,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=0, keepdims=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 0, 0)[0]  # axis=0, keepdims=0
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 2: 2D 张量，轴 0 (keepdims=True)
print("Case 2: 2D 张量，轴 0 (keepdims=True)")
numpy_in = np.random.uniform(-8, 8, (1024, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=0, keepdims=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 0, 1)[0]  # axis=0, keepdims=1
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 3: 2D 张量，轴 0 (keepdims=False)
print("Case 3: 2D 张量，轴 0 (keepdims=False)")
numpy_in = np.random.uniform(-8, 8, (1024, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=0, keepdims=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 0, 0)[0]  # axis=0, keepdims=0
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 4: 2D 张量，轴 1 (keepdims=True)
print("Case 4: 2D 张量，轴 1 (keepdims=True)")
numpy_in = np.random.uniform(-8, 8, (1024, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=1, keepdims=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 1, 1)[0]  # axis=1, keepdims=1
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 5: 2D 张量，轴 1 (keepdims=False)
print("Case 5: 2D 张量，轴 1 (keepdims=False)")
numpy_in = np.random.uniform(-8, 8, (1024, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=1, keepdims=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 1, 0)[0]  # axis=1, keepdims=0
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 6: 2D 张量，负数轴索引
print("Case 6: 2D 张量，负数轴索引")
numpy_in = np.random.uniform(-8, 8, (1024, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=-1, keepdims=True)
print("NumPy (axis=-1):", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, -1, 1)[0]  # axis=-1, keepdims=1
print(f"{argmin_op} (axis=-1):", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 7: 3D 张量，轴 0
print("Case 7: 3D 张量，轴 0")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=0, keepdims=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 0, 1)[0]  # axis=0, keepdims=1
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 8: 3D 张量，轴 1
print("Case 8: 3D 张量，轴 1")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=1, keepdims=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 1, 1)[0]  # axis=1, keepdims=1
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 9: 3D 张量，轴 2
print("Case 9: 3D 张量，轴 2")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=2, keepdims=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 2, 1)[0]  # axis=2, keepdims=1
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()

# Case 10: 3D 张量，不保持维度
print("Case 10: 3D 张量，不保持维度")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=1, keepdims=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = argmin_op.run(numpy_in, 1, 0)[0]  # axis=1, keepdims=0
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.array_equal(numpy_out, kp_out))
print()