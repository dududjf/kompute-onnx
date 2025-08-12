from kp import Manager
import numpy as np
import time
from kp_onnx.kop_cast_like import CastLikeOp

device_id = 0
mgr = Manager(device_id)
print("Using device:", mgr.list_devices()[device_id])

cast_like_op = CastLikeOp(mgr, ['input', 'like'], ['output'])


def run_test(name, src, like):
    print(f"\n== {name} ==")

    # Numpy 期望输出
    numpy_out = src.astype(like.dtype, copy=False)

    # warmup（建图编译）
    _ = cast_like_op.run(src, like)

    # 单次 Numpy 执行计时
    t0 = time.time()
    numpy_result = src.astype(like.dtype, copy=False)
    numpy_time = time.time() - t0
    print(f"Numpy time: {numpy_time:.6f} seconds")

    # 单次 Kompute 执行计时
    t1 = time.time()
    kp_result = cast_like_op.run(src.copy(), like)[0]
    kp_time = time.time() - t1
    print(f"Kompute time: {kp_time:.6f} seconds")

    # 精度验证
    if like.dtype == np.int32:
        ok = np.array_equal(numpy_out, kp_result)
        max_err = (numpy_out != kp_result).sum()
    else:
        ok = np.allclose(numpy_out, kp_result, rtol=1e-6, atol=0.0)
        max_err = np.max(np.abs(numpy_out.astype(np.float32) - kp_result.astype(np.float32))) if kp_result.size else 0.0

    print(f"Max error: {max_err}")
    print(f"All close: {ok}")


# 测试 1: float32 -> float32
src = np.random.random((64, 1024)).astype(np.float32)
like = np.empty_like(src, dtype=np.float32)
run_test("Test1-f32->f32-64x1024", src, like)

# 测试 2: float32 -> int32
src = (np.random.random((64, 1024)).astype(np.float32) * 10.0 - 5.0)
like = np.empty_like(src, dtype=np.int32)
run_test("Test2-f32->i32-64x1024", src, like)

# 测试 3: int32 -> float32
src = np.random.randint(-1000, 1000, size=(64, 1024)).astype(np.int32)
like = np.empty_like(src, dtype=np.float32)
run_test("Test3-i32->f32-64x1024", src, like)

# 测试 4: 更高维输入（f32->i32）
src = (np.random.random((2, 5, 1000, 512)).astype(np.float32) * 20.0 - 10.0)
like = np.empty_like(src, dtype=np.int32)
run_test("Test4-f32->i32-2x5x1000x512", src, like)

# 测试 5: 大规模输入（1e8 元素）
large_size = int(1e8)
src = (np.random.random(large_size).astype(np.float32) * 10.0)
like = np.empty_like(src, dtype=np.int32)
run_test(f"Test5-f32->i32-{large_size}-elements", src, like)
