from kp import Manager
import numpy as np
import time
from kp_onnx.kop_constant_of_shape import ConstantOfShapeOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
constant_of_shape_op = ConstantOfShapeOp(mgr)

print('Case 1')
shape = np.array([2, 3], dtype=np.int64)
value = np.array([5.0], dtype=np.float32)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.float32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# 未提供常量值时，应生成 float32 默认 0
print('Case 2')
shape = np.array([1, 4, 2], dtype=np.int64)

start_time = time.time()
numpy_out = np.full(tuple(shape), 0.0, dtype=np.float32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
shape = np.array([3, 2], dtype=np.int64)
value = np.array([2.5], dtype=np.float32)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.float32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
shape = np.array([100, 100], dtype=np.int64)
value = np.array([1.0], dtype=np.float32)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.float32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
shape = np.array([2, 3, 4, 5], dtype=np.int64)
value = np.array([-1.5], dtype=np.float32)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.float32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 6')
shape = np.array([3, 3], dtype=np.int64)
value = np.array([10], dtype=np.int32)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.int32)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 7')
shape = np.array([2, 2], dtype=np.int64)
value = np.array([5000000000], dtype=np.int64)

start_time = time.time()
numpy_out = np.full(tuple(shape), value[0], dtype=np.int64)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = constant_of_shape_op.run(shape, value)[0]
print(f"{constant_of_shape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))
