from kp import Manager
import numpy as np
import time
from kp_onnx.kop_cos import CosOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

cos_op = CosOp(mgr, ['input'], ['output'])

numpy_in = np.random.uniform(-100, 100, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out = np.cos(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = cos_op.run(numpy_in)[0]
print(f"{cos_op}:", time.time() - start_time, "seconds")

print("numpy_out:", numpy_out)
print("kp_out:", kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))