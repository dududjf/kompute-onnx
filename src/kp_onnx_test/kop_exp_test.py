from kp import Manager
import numpy as np
import time
from kp_onnx.kop_exp import ExpOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

exp_op = ExpOp(mgr, ['input'], ['output'])

# ===== 测试 1：中等规模 2D =====
numpy_in = np.random.uniform(-5, 5, (64, 1024)).astype(np.float32)

start_time = time.time()
numpy_out = np.exp(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = exp_op.run(numpy_in)[0]
print(f"{exp_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# ===== 测试 2：高维 4D =====
numpy_in = np.random.uniform(-3, 3, (2, 5, 1000, 512)).astype(np.float32)

start_time = time.time()
numpy_out = np.exp(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = exp_op.run(numpy_in)[0]
print(f"{exp_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
