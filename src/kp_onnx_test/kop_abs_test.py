from kp import Manager
import numpy as np
import time
from kp_onnx.kop_abs import AbsOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

abs_op = AbsOp(mgr, ['input'], ['output'])
x = np.random.uniform(-5, 5, size=(64, 1024)).astype(np.float32)

t0 = time.time()
numpy_out = np.abs(x)
print("ABS Numpy:", time.time() - t0, "seconds")

t0 = time.time()
kp_out = abs_op.run(x)[0]
print(f"{abs_op}:", time.time() - t0, "seconds")
print("ABS Max error:", np.abs(numpy_out - kp_out).max())
print("ABS All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))