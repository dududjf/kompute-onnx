from kp import Manager
import numpy as np
import time
from kp_onnx.kop_sigmoid import SigmoidOp


def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        return np.array(1.0 / (1.0 + np.exp(-x)) if x > 0 else np.exp(x) / (1.0 + np.exp(x)), dtype=x.dtype)
    if x.size == 0:
        return x
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sigmoid_op = SigmoidOp(mgr)

numpy_in = np.random.uniform(-1000, 1000, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out = stable_sigmoid(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sigmoid_op.run(numpy_in)[0]
print(f"{sigmoid_op}:", time.time() - start_time, "seconds")

print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
