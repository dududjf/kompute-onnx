from kp import Manager
import numpy as np
import time
from kp_onnx.kop_cos import CosOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

cos_op = CosOp(mgr, ['input'], ['output'])

#浮点数
numpy_in = np.random.random(1024 * 1024 * 16)

start_time = time.time()
numpy_out = np.cos(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = cos_op.run(numpy_in)[0]
print(f"{cos_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out)
print("kp_out:", kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

#整数
int_in = np.random.randint(-10, 10, 1024 * 1024 * 16, dtype=np.int32)

start_time = time.time()
int_numpy_out = np.cos(int_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
int_kp_out = cos_op.run(int_in)[0]
print(f"{cos_op}:", time.time() - start_time, "seconds")

print("int_numpy_out:", int_numpy_out)
print("int_kp_out:", int_kp_out)
print(np.allclose(int_numpy_out, int_kp_out, rtol=1e-4, atol=1e-4))

#布尔型
bool_in = np.random.choice([True, False], 1024 * 1024 * 16)

start_time = time.time()
bool_numpy_out = np.cos(bool_in.astype(np.float32))
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
bool_kp_out = cos_op.run(bool_in.astype(np.float32))[0]
print(f"{cos_op}:", time.time() - start_time, "seconds")

print("bool_numpy_out:", bool_numpy_out)
print("bool_kp_out:", bool_kp_out)
print(np.allclose(bool_numpy_out, bool_kp_out, rtol=1e-4, atol=1e-4))
