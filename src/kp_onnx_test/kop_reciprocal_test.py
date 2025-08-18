from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_reciprocal import ReciprocalOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reciprocal_op = ReciprocalOp(mgr, ['input'], ['output'])

x = np.random.random((10240, 10240))

start_time = time.time()
np_out = np.reciprocal(x).astype(np.float32)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = reciprocal_op.run(x)[0]
print(f"{reciprocal_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
