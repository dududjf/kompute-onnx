from kp import Manager
import numpy as np
import time
from kp_onnx.kop_tanh import TanhOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

tanh_op = TanhOp(mgr, ['input'], ['output'])

numpy_in = np.random.uniform(-1000, 1000, 1024 * 1024 * 16)

start_time = time.time()
numpy_out = np.tanh(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = tanh_op.run(numpy_in)[0]
print(f"{tanh_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out)
print("kp_out:", kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))