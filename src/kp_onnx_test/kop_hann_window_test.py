from kp import Manager
import numpy as np
import time
from kp_onnx.kop_hann_window import HannWindowOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
hann_op = HannWindowOp(mgr)

def numpy_hann_reference(size, periodic=False, dtype=np.float32):
    if size == 0:
        return np.array([], dtype=dtype)
    elif size == 1:
        return np.array([0.], dtype=dtype)
    else:
        N_1 = size if periodic else (size - 1)
        ni = np.arange(size)
        return (np.sin(ni * np.pi / N_1) ** 2).astype(dtype)

print("\nCase 1: size=0, periodic=False")
size, periodic, dtype = 0, False, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 2: size=0, periodic=True")
size, periodic, dtype = 0, True, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 3: size=1, periodic=False")
size, periodic, dtype = 1, False, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 4: size=1, periodic=True")
size, periodic, dtype = 1, True, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 5: size=8, periodic=False")
size, periodic, dtype = 8, False, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 6: size=8, periodic=True")
size, periodic, dtype = 8, True, np.float32
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 7: size=16, periodic=False, dtype=float64")
size, periodic, dtype = 16, False, np.float64
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print("\nCase 8: size=16, periodic=True, dtype=float16")
size, periodic, dtype = 16, True, np.float16
start_time = time.time()
numpy_out = numpy_hann_reference(size, periodic, dtype)
print("Numpy:", numpy_out, "time:", time.time() - start_time, "seconds")
start_time = time.time()
kp_out = hann_op.run(size, 1.0 if periodic else 0.0, dtype=dtype)[0]
print(f"{hann_op}:", kp_out, "time:", time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))
