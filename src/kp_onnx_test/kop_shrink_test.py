from kp import Manager
import numpy as np
import time
from kp_onnx.kop_shrink import ShrinkOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

shrink_op = ShrinkOp(mgr)


def numpy_shrink(data, lam=0.5, bis=0.0):
    return np.where(
        data < -lam,
        data + bis,
        np.where(data > lam, data - bis, 0.0)
    )


x = np.random.random((1024, 1024)).astype(np.float32)


# -------- Case 1: lambda: None, bias: None --------
print("Case 1: Shrink with lambda: None, bias: None")

start_time = time.time()
np_out = numpy_shrink(x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = shrink_op.run(x)[0]
print(f"{shrink_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: lambda: 1.2, bias: None --------
print("Case 2: Shrink with lambda: 1.2, bias: None")

lambd = float(1.2)

start_time = time.time()
np_out = numpy_shrink(x, lambd)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = shrink_op.run(x, lambd)[0]
print(f"{shrink_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: lambda: 1.2, bias: 0.3 --------
print("Case 2: Shrink with lambda: 1.2, bias: 0.3")

lambd = float(1.2)
bias = float(0.3)

start_time = time.time()
np_out = numpy_shrink(x, lambd, bias)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = shrink_op.run(x, lambd, bias)[0]
print(f"{shrink_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
