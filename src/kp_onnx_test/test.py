# mini_mean_debug.py
from kp import Manager
import numpy as np
import kp_onnx.kop_mean as km  # 你的 MeanOp 所在模块

np.random.seed(0)
mgr = Manager(0)
op = km.MeanOp(mgr)

# 构造强广播：a (2,2,3) vs b (3,2,2,1) vs c (1,1,1)  三输入
a = np.arange(12, dtype=np.float32).reshape(2,2,3)       # [ [0..2],[3..5] ; [6..8],[9..11] ]
b = (np.arange(3*2*2, dtype=np.float32).reshape(3,2,2,1) + 100.)   # 每个 z 维为1，要沿 z 广播
c = np.array([[[[0.5]]]], dtype=np.float32)              # 全局常数，形状(1,1,1,1)

# 期望输出形状
out_shape = np.broadcast(a.reshape(1,2,2,3), b, c).shape
print("out_shape =", out_shape)

# NumPy 基准
np_sum  = a.reshape(1,2,2,3) + b + c
np_mean = (np_sum / 3.0)

# ---- 改一版 MeanOp.run，让它能返回“中间sum结果”（仅用于定位，别永久保存）----
# 你可以临时在 MeanOp 里加一个调试方法；这里我们直接调用 run，然后自己再把 acc/out_mean 打印出来。
# 为了简单，直接调用 op.run 并对照最终 mean；若不一致，再在 MeanOp.run 里：
#   1) 在最后 scale 核之前，加 seq_sum_sync.record(kp.OpTensorSyncLocal([acc])) .eval()
#   2) 打印 acc.data().reshape(out_shape) 与 np_sum 对比

kp_mean = op.run(a, b, c)[0]

print("\n=== NumPy SUM ===")
print(np_sum)
print("\n=== NumPy MEAN ===")
print(np_mean)

print("\n=== Kompute MEAN ===")
print(kp_mean)

mask = ~np.isclose(np_mean, kp_mean, rtol=1e-6, atol=1e-6)
idxs = np.argwhere(mask)
print("\nMismatch count:", idxs.shape[0])
if idxs.size:
    print("First mismatches (up to 5):")
    for idx in idxs[:5]:
        idx = tuple(idx.tolist())
        print(f"idx={idx} | np={np_mean[idx]:.6f}, kp={kp_mean[idx]:.6f}")