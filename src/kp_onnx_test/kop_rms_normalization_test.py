from kp import Manager
import numpy as np
import time
from kp_onnx.kop_rms_normalization import RmsNormalizationOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

rms_normalization_op = RmsNormalizationOp(mgr)


def np_rms_norm(X, Scale, axis=-1, epsilon=1e-05, stash_type=1):
    if stash_type != 1:
        raise NotImplementedError(
            f"RMSNormalization not implemented for stash_type={stash_type} != 1."
        )
    shape = X.shape
    rank = len(shape)
    if axis < 0:
        axis = axis + rank

    # This computes RMS for every x_mat's column.
    x_squared = np.power(X, 2)
    x_squared_mean = np.mean(x_squared, axis=tuple(range(axis, len(shape))), keepdims=True)
    # epsilon adjustment to avoid divide-by-zero.
    rmseps = x_squared_mean + epsilon

    rms = np.sqrt(rmseps)
    rms_reciprocal = np.reciprocal(rms)
    print("rms_reciprocal shape:", rms_reciprocal.shape)

    y_mat = X * rms_reciprocal
    Y = y_mat * Scale

    return Y.astype(X.dtype)


x = np.random.random((255, 511, 1023)).astype(np.float32)

# -------- Case 1: data: 3D --------
print("Case 1: data: 3D")
scale = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
np_out = np_rms_norm(x, scale)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
rms_normalization_op.axis = -1
rms_normalization_op.epsilon = 1e-05
rms_normalization_op.stash_type = 1
kp_out = rms_normalization_op.run(x, scale)[0]
print(f"{rms_normalization_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: data: 3D, axis: 0, epsilon: 1e-03 --------
print("Case 2: data: 3D, axis: 0, epsilon: 1e-03")
scale = np.random.random((255, 511, 1)).astype(np.float32)

start_time = time.time()
np_out = np_rms_norm(x, scale, axis=0, epsilon=1e-03, stash_type=1)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
rms_normalization_op.axis = 0
rms_normalization_op.epsilon = 1e-03
rms_normalization_op.stash_type = 1
kp_out = rms_normalization_op.run(x, scale)[0]
print(f"{rms_normalization_op}: ", time.time() - start_time, "seconds")

print("x_shape:", x.shape)
print("np_out_shape: ", np_out.shape)
print("kp_out_shape: ", kp_out.shape)

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")