from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_identity import IdentityOp

# Device info
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

identity_op = IdentityOp(mgr, ['data'], ['output'])

print("Case 1: identity on 2D array")
x = np.random.random((10240, 10240))

start_time = time.time()
np_out = x.copy().astype(np.float32)
print("cpu:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = identity_op.run(x)[0]
print(f"{identity_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
