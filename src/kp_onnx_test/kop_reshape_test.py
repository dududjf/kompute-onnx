from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reshape import ReshapeOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
reshape_op = ReshapeOp(mgr)

#Case 1
numpy_in = np.random.random((2, 3, 4))
shape = np.array([4, 6])
start_time = time.time()
numpy_out = np.reshape(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

#Case 2: 含 -1 自动推断
numpy_in = np.random.random((2, 3, 4))
shape = np.array([-1, 4])
start_time = time.time()
numpy_out = np.reshape(numpy_in, (6, 4))
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

#Case 3: 含 0, allowzero=0（替换为输入维度）
numpy_in = np.random.random((2, 3, 4))
shape = np.array([2, 0, 4])
start_time = time.time()
numpy_out = np.reshape(numpy_in, (2, 3, 4))
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reshape_op.run(numpy_in, shape, np.array([0]))[0]  # allowzero=0
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
