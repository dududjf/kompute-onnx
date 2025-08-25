from kp import Manager
import numpy as np
import time
from kp_onnx.kop_leaky_relu import LeakyReluOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

leaky_relu_op = LeakyReluOp(mgr)

# ---------------- Case 1: alpha: None ----------------
print("Case 1: LeakyReLU alpha: None")
x = np.random.random((1024, 1024)).astype(np.float32)
print("Input shape:", x.shape)

start_time = time.time()
np_out = np.where(x >= 0.0, x, 0.01 * x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = leaky_relu_op.run(x)[0]
print(f"{leaky_relu_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: alpha: 0.2 ----------------
print("Case 2: LeakyReLU alpha=0.2")
x = np.random.random((1024, 1024))
alpha = 0.2
print("Input shape:", x.shape)

start_time = time.time()
np_out = np.where(x >= 0.0, x, alpha * x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = leaky_relu_op.run(x, alpha)[0]
print(f"{leaky_relu_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


