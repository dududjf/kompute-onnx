from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reshape import ReshapeOp


def reshape_reference(data: np.ndarray, shape: np.ndarray, allowzero: int = 0):
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
reshape_op = ReshapeOp(mgr)

# Case 1: 普通 reshape
numpy_in = np.random.random((2, 3, 4))
shape = np.array([4, 6])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 2: 含 -1 自动推导
numpy_in = np.random.random((2, 3, 4))
shape = np.array([-1, 4])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 3: 含 0, allowzero=0
numpy_in = np.random.random((2, 3, 4))
shape = np.array([0, 12])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape, allowzero=0)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape, 0)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 4: 含 0, allowzero=1
numpy_in = np.random.random((2, 3, 4))
shape = np.array([2, 12])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape, allowzero=1)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape, 1)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 5: 标量 tensor
numpy_in = np.array(42)
shape = np.array([1])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 6: 多维 reshape
numpy_in = np.random.random((2, 3, 4, 5))
shape = np.array([4, 5, 6])

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 7: allowzero=0, 使用实例属性
numpy_in = np.random.random((2, 3, 4))
shape = np.array([0, 12])
reshape_op.allowzero = 0

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape, allowzero=reshape_op.allowzero)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 8: allowzero=1, 使用实例属性
numpy_in = np.random.random((2, 3, 4))
shape = np.array([2, 12])
reshape_op.allowzero = 1

start_time = time.time()
numpy_out = reshape_reference(numpy_in, shape, allowzero=reshape_op.allowzero)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")
