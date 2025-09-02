from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reshape import ReshapeOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
reshape_op = ReshapeOp(mgr)

def onnx_reshape(data: np.ndarray, shape: np.ndarray, allowzero: int = 0) -> np.ndarray:
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped

print('Case 1')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.array([1023, 15], dtype=np.int64)
start_time = time.time()
numpy_out = onnx_reshape(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = reshape_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{reshape_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Allclose:", np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

#-1 自动推算
print('Case 2')
numpy_in_1 = np.random.random((6, 5, 4))
numpy_in_2 = np.array([3, -1, 4], dtype=np.int64)
numpy_out = onnx_reshape(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape)
kp_out = reshape_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{reshape_op}:", kp_out.shape)
print("Allclose:", np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

#allowzero=0
print('Case 3')
numpy_in_1 = np.random.random((2, 3, 4))
numpy_in_2 = np.array([0, 3, 4], dtype=np.int64)
numpy_in_3 = np.array([0], dtype=np.int64)
numpy_out = onnx_reshape(numpy_in_1, numpy_in_2, allowzero=int(numpy_in_3[0]))
print("Numpy:", numpy_out.shape)
kp_out = reshape_op.run(numpy_in_1, numpy_in_2, numpy_in_3)[0]
print(f"{reshape_op}:", kp_out.shape)
print("Allclose:", np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))
