from kp import Manager
import numpy as np
import time
from kp_onnx.kop_sub import SubOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

# case 1
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((3, 1023, 1023, 1))

start_time = time.time()
numpy_out = numpy_in_1 - numpy_in_2
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

sub_op = SubOp(mgr, ['input1', 'input2'], ['output'])
start_time = time.time()
kp_out = sub_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sub_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# case 2
numpy_in_1 = np.random.random((3, 1023, 1023, 1))
numpy_in_2 = np.random.random((1023, 15))

start_time = time.time()
numpy_out = numpy_in_1 - numpy_in_2
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

sub_op = SubOp(mgr, ['input1', 'input2'], ['output'])
start_time = time.time()
kp_out = sub_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sub_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# case 3
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((1023, 1))

start_time = time.time()
numpy_out = numpy_in_1 - numpy_in_2
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

sub_op = SubOp(mgr, ['input1', 'input2'], ['output'])
start_time = time.time()
kp_out = sub_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sub_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# case 4
numpy_in_1 = np.random.random((1023,))
numpy_in_2 = np.random.random((1,))

start_time = time.time()
numpy_out = numpy_in_1 - numpy_in_2
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

sub_op = SubOp(mgr, ['input1', 'input2'], ['output'])
start_time = time.time()
kp_out = sub_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sub_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))
