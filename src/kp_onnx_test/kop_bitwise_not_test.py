from kp import Manager
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kp_onnx.kop_bitwise_not import BitwiseNotOp


device_id = 0
mgr = Manager(device_id)
print(mgr.list_devices()[device_id])

bitwise_not_op = BitwiseNotOp(mgr, ['input'], ['output'])
numpy_in = np.random.randint(-1000, 1000, 1024 * 1024)

start_time = time.time()
numpy_out = np.bitwise_not(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = bitwise_not_op.run(numpy_in)[0]
print(f"{bitwise_not_op}:", time.time() - start_time, "seconds")

print(numpy_out)
print(kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))