from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_clip import ClipOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

clip_op = ClipOp(mgr, ['data', 'min', 'max'], ['output'])


# Unified numpy reference implementation (can handle None)
def numpy_clip_like(x, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        return x
    if vmin is None:
        return np.minimum(x, vmax)
    if vmax is None:
        return np.maximum(x, vmin)
    return np.clip(x, vmin, vmax)


x = np.random.random(32 * 1024 * 1024)

# -------- Case 1: min: None, max: None --------
print("Case 1: min=None, max=None")
start_time = time.time()
np_out = numpy_clip_like(x, None, None).astype(np.float32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = clip_op.run(x)[0]
print(f"{clip_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: only min --------
print("Case 2: min=0.2, max=None")
vmin = np.asarray(0.2, dtype=np.float32)
start_time = time.time()
np_out = numpy_clip_like(x, vmin, None).astype(np.float32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = clip_op.run(x, vmin)[0]
print(f"{clip_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: min & max --------
print("Case 3: min=0.2, max=0.7")
vmin = np.asarray(0.2, dtype=np.float32)
vmax = np.asarray(0.7, dtype=np.float32)
start_time = time.time()
np_out = numpy_clip_like(x, vmin, vmax).astype(np.float32)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = clip_op.run(x, vmin, vmax)[0]
print(f"{clip_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
