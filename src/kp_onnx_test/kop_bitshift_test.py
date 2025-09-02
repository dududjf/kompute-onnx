from kp import Manager
import numpy as np
import time
from kp_onnx.kop_bitshift import BitShiftOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

bitshift_left_i = BitShiftOp(mgr, direction="LEFT")
print("Case 1")
a = np.random.randint(0, 100, (1023, 15), dtype=np.int32)
b = np.random.randint(0, 5, (3, 1023, 1023, 1), dtype=np.int32)
start = time.time()
numpy_out = np.left_shift(a, b)
print("Numpy:", numpy_out.shape, time.time() - start, "seconds")
start = time.time()
kp_out = bitshift_left_i.run(a, b)[0]
print(f"{bitshift_left_i}:", kp_out.shape, time.time() - start, "seconds")
print("Match:", np.array_equal(numpy_out, kp_out))

print("Case 2")
a = np.random.randint(-50, 50, (3, 1023, 1023, 1), dtype=np.int32)
b = np.random.randint(0, 5, (1023, 15), dtype=np.int32)
start = time.time()
numpy_out = np.left_shift(a, b)
print("Numpy:", numpy_out.shape, time.time() - start, "seconds")
start = time.time()
kp_out = bitshift_left_i.run(a, b)[0]
print(f"{bitshift_left_i}:", kp_out.shape, time.time() - start, "seconds")
print("Match:", np.array_equal(numpy_out, kp_out))

bitshift_right_i = BitShiftOp(mgr, direction="RIGHT")
print("Case 3")
a = np.random.randint(-500, 500, (225, 255, 15), dtype=np.int32)
b = np.random.randint(0, 5, (3, 3, 225, 255, 1), dtype=np.int32)
start = time.time()
numpy_out = np.right_shift(a, b)
print("Numpy:", numpy_out.shape, time.time() - start, "seconds")
start = time.time()
kp_out = bitshift_right_i.run(a, b)[0]
print(f"{bitshift_right_i}:", kp_out.shape, time.time() - start, "seconds")
print("Match:", np.array_equal(numpy_out, kp_out))
