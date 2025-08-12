from kp import Manager
import numpy as np
import time
from kp_onnx.kop_neg import NegOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

neg_op = NegOp(mgr, ['input'], ['output'])

# ===== 测试 1: 随机数据 =====
numpy_in = np.random.randn(2, 5, 1000, 512).astype(np.float32)

# Numpy 实现
start_time = time.time()
numpy_out = -numpy_in
print("Numpy:", time.time() - start_time, "seconds")

# Kompute 实现
start_time = time.time()
kp_out = neg_op.run(numpy_in)[0]
print(f"{neg_op}:", time.time() - start_time, "seconds")

# 精度验证
print('Max error:', np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# ===== 测试 2: 大规模输入 =====
numpy_in = np.random.randn(4096, 4096).astype(np.float32)

start_time = time.time()
numpy_out = -numpy_in
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = neg_op.run(numpy_in)[0]
print(f"{neg_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
