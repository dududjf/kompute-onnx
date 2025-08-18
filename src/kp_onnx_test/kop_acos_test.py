# kp_onnx_test/kop_acos_test.py
from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_acos import AcosOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

acos_op = AcosOp(mgr, ['input'], ['output'])

print("Case 1: acos on 2D array")
x = np.random.uniform(-1.0, 1.0, (4096, 4096))

# NumPy
start_time = time.time()
np_out = np.arccos(x).astype(np.float32)
print("Numpy:", time.time() - start_time, "seconds")

# AcosOp
start_time = time.time()
kp_out = acos_op.run(x)[0]
print(f"{acos_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
