from kp import Manager
import numpy as np
import time
from kp_onnx.kop_sinh import SinhOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sinh_op = SinhOp(mgr, ['input'], ['output'])

#浮点数
numpy_in_f32 = np.random.uniform(-80, 80, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out_f32 = np.sinh(numpy_in_f32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_f32 = sinh_op.run(numpy_in_f32)[0]
print(f"{sinh_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out_f32)
print("kp_out:", kp_out_f32)
print(np.allclose(numpy_out_f32, kp_out_f32, rtol=1e-4, atol=1e-4))

#整数
int_in = np.random.randint(-80, 80, 1024 * 1024 * 16)

start_time = time.time()
int_numpy_out = np.sinh(int_in.astype(np.float32))
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
int_kp_out = sinh_op.run(int_in)[0]
print(f"{sinh_op}:", time.time() - start_time, "seconds")

print("int_numpy_out:", int_numpy_out)
print("int_kp_out:", int_kp_out)
print(np.allclose(int_numpy_out, int_kp_out, rtol=1e-4, atol=1e-4))

#布尔型
bool_in = np.random.choice([True, False], 1024 * 1024 * 16)

start_time = time.time()
bool_numpy_out = np.sinh(bool_in.astype(np.float32))
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
bool_kp_out = sinh_op.run(bool_in.astype(np.float32))[0]
print(f"{sinh_op}:", time.time() - start_time, "seconds")

print("bool_numpy_out:", bool_numpy_out)
print("bool_kp_out:", bool_kp_out)
print(np.allclose(bool_numpy_out, bool_kp_out, rtol=1e-4, atol=1e-4))

