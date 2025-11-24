from kp import Manager
import numpy as np
import time
from kp_onnx.kop_matmul_integer import MatMulIntegerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
matmul_integer_op = MatMulIntegerOp(mgr)


def np_matmul_integer(A, B, a_zero_point=None, b_zero_point=None):  # type: ignore
    A32 = A.astype(np.int32)
    if a_zero_point is not None:
        A32 -= a_zero_point
    B32 = B.astype(np.int32)
    if b_zero_point is not None:
        B32 -= b_zero_point
    return A32 @ B32


print('Case 1')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}: ", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 2')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.uint8)
numpy_in_2 = np.random.random((2, 5, 512, 1024)).astype(np.uint8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 3')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = np.random.random((2, 5, 1000, 1)).astype(np.int8)
b_zero_point = np.random.random((1024,)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 4')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.uint8)
numpy_in_2 = np.random.random((2, 5, 512, 1024)).astype(np.uint8)
a_zero_point = np.random.random((2, 5, 1000, 1)).astype(np.uint8)
b_zero_point = np.random.random((2, 5, 1, 1024)).astype(np.uint8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 5')
numpy_in_1 = np.random.random((1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = np.random.random((1000,)).astype(np.int8)
b_zero_point = np.random.random((1024,)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

print('Case 6')
numpy_in_1 = np.random.random((1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = 2
b_zero_point = 5

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))