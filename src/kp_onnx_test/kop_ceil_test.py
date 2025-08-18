from kp import Manager
import numpy as np
import time
from kp_onnx.kop_ceil import CeilOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

ceil_op = CeilOp(mgr, ['input'], ['output'])

#浮点数
numpy_in_f32 = np.random.uniform(-1000, 1000, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out_f32 = np.ceil(numpy_in_f32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_f32 = ceil_op.run(numpy_in_f32)[0]
print(f"{ceil_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out_f32)
print("kp_out:", kp_out_f32)
print(np.allclose(numpy_out_f32, kp_out_f32, rtol=1e-4, atol=1e-4))

#整数
numpy_in_i32 = np.random.randint(-1000, 1000, 1024 * 1024 * 16)

start_time = time.time()
numpy_out_i32 = np.ceil(numpy_in_i32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_i32 = ceil_op.run(numpy_in_i32)[0]
print(f"{ceil_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out_i32)
print("kp_out:", kp_out_i32)
print(np.allclose(numpy_out_i32, kp_out_i32, rtol=1e-4, atol=1e-4))

#布尔型
numpy_in_bool = np.random.choice([True, False], 1024 * 1024 * 16)

start_time = time.time()
numpy_out_bool = np.ceil(numpy_in_bool.astype(np.float32))
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_bool = ceil_op.run(numpy_in_bool.astype(np.float32))[0]
print(f"{ceil_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out_bool)
print("kp_out:", kp_out_bool)
print(np.allclose(numpy_out_bool, kp_out_bool, rtol=1e-4, atol=1e-4))


