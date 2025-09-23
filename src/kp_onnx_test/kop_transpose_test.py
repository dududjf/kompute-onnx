from kp import Manager
import numpy as np
import time
from kp_onnx.kop_transpose import TransposeOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

transpose_op = TransposeOp(mgr)


def onnx_transpose(x: np.ndarray, perm=None) -> np.ndarray:
    return np.transpose(x, perm)

# Case 1: 默认(逆序）
print("\nCase 1: default")
x = np.random.uniform(-10, 10, (256, 128, 64, 32)).astype(np.float32)

t0 = time.time()
numpy_out = onnx_transpose(x)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = transpose_op.run(x)[0]
print(f"{transpose_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 2: 指定 perm
print("\nCase 2: perm=(1, 3, 0, 2)")
x = np.random.uniform(-10, 10, (256, 128, 64, 32)).astype(np.float32)

perm = (1, 3, 0, 2)
t0 = time.time()
numpy_out = onnx_transpose(x, perm)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = transpose_op.run(x, np.array(perm, dtype=np.int32))[0]
print(f"{transpose_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 3: 指定带负值perm
print("\nCase 3: perm=(-1, -3, 0, -2)")
x = np.random.uniform(-10, 10, (256, 128, 64, 32)).astype(np.float32)

perm = (-1, -3, 0, -2)
t0 = time.time()
numpy_out = onnx_transpose(x, perm)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = transpose_op.run(x, np.array(perm, dtype=np.int32))[0]
print(f"{transpose_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
