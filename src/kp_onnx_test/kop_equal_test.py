from kp import Manager
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kp_onnx.kop_equal import EqualOp

device_id = 0
mgr = Manager(device_id)
print(mgr.list_devices()[device_id])

equal_op = EqualOp(mgr, ['input_a', 'input_b'], ['output'])

numpy_in_a = np.random.random(1024 * 1024)
numpy_in_b = np.random.random(1024 * 1024)


start_time = time.time()
numpy_out = np.equal(numpy_in_a, numpy_in_b)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = equal_op.run(numpy_in_a, numpy_in_b)[0]
print(f"{equal_op}:", time.time() - start_time, "seconds")

print(numpy_out)
print(kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print(np.array_equal(numpy_out, kp_out))
