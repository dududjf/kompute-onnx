from kp import Manager
import numpy as np
import time
from kp_onnx.kop_relu import ReluOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

relu_op = ReluOp(mgr, ['input'], ['output'])

# ====== 测试 1：中等规模二维输入 ======
numpy_in = np.random.randn(64, 1024).astype(np.float32)

start_time = time.time()
numpy_out = np.maximum(numpy_in, 0.0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = relu_op.run(numpy_in)[0]
print(f"{relu_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# ====== 测试 2：高维输入 ======
numpy_in = np.random.randn(2, 5, 1000, 512).astype(np.float32)

start_time = time.time()
numpy_out = np.maximum(numpy_in, 0.0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = relu_op.run(numpy_in)[0]
print(f"{relu_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
