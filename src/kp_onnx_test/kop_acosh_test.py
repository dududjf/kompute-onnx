from kp import Manager
import numpy as np
import time
from kp_onnx.kop_acosh import AcoshOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

acosh_op = AcoshOp(mgr, ['input'], ['output'])

#浮点数
numpy_in = np.random.uniform(1, 1000, 1024 * 1024 * 16)

start_time = time.time()
numpy_out = np.arccosh(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = acosh_op.run(numpy_in)[0]
print(f"{acosh_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out)
print("kp_out:", kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

#整数
int_in = np.random.randint(1, 1000, 1024 * 1024 * 16)

start_time = time.time()
int_numpy_out = np.arccosh(int_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
int_kp_out = acosh_op.run(int_in)[0]
print(f"{acosh_op}:", time.time() - start_time, "seconds")

print("int_numpy_out:", int_numpy_out)
print("int_kp_out:", int_kp_out)
print(np.allclose(int_numpy_out, int_kp_out, rtol=1e-4, atol=1e-4))

