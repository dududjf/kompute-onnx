from kp import Manager
import numpy as np
import time
from kp_onnx.kop_asin import AsinOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

asin_op = AsinOp(mgr, ['input'], ['output'])

print("Case 1: asin on 2D array")
x = np.random.uniform(-1.0, 1.0, (4096, 4096)).astype(np.float32)

# NumPy
start_time = time.time()
np_out = np.arcsin(x)
print("Numpy:", time.time() - start_time, "seconds")

# AsinOp
start_time = time.time()
kp_out = asin_op.run(x)[0]
print(f"{asin_op}: ", time.time() - start_time, "seconds")

print("shape equal: ", kp_out.shape == np_out.shape)
print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
