"""
Test script for kop_einsum_new.py
Tests various einsum patterns to verify the decomposition-based implementation
"""
import numpy as np
import time
import kp

from kp_onnx.kop_einsum import EinsumOp


def onnx_einsum(equation, *inputs):
    """Reference implementation using numpy"""
    return np.einsum(equation, *inputs)


# Initialize Kompute
mgr = kp.Manager()
print(mgr.get_device_properties())
print()

# Test Case 1: Matrix multiplication (ij,jk->ik)
print("=" * 60)
print("Test Case 1: Matrix multiplication (ij,jk->ik)")
print("=" * 60)
equation = "ij,jk->ik"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 2: Batch matrix multiplication (ijk,ikl->ijl)
print("=" * 60)
print("Test Case 2: Batch matrix multiplication (ijk,ikl->ijl)")
print("=" * 60)
equation = "ijk,ikl->ijl"
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 3: Outer product (i,j->ij)
print("=" * 60)
print("Test Case 3: Outer product (i,j->ij)")
print("=" * 60)
equation = "i,j->ij"
a = np.random.randn(3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 4: Transpose (ij->ji)
print("=" * 60)
print("Test Case 4: Transpose (ij->ji)")
print("=" * 60)
equation = "ij->ji"
a = np.random.randn(3, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 5: Sum reduction (ij->i)
print("=" * 60)
print("Test Case 5: Sum reduction (ij->i)")
print("=" * 60)
equation = "ij->i"
a = np.random.randn(3, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 6: Dot product (i,i->)
print("=" * 60)
print("Test Case 6: Dot product (i,i->)")
print("=" * 60)
equation = "i,i->"
a = np.random.randn(5).astype(np.float32)
b = np.random.randn(5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 7: Element-wise multiply with reduction (ij,ij->i)
print("=" * 60)
print("Test Case 7: Element-wise multiply with reduction (ij,ij->i)")
print("=" * 60)
equation = "ij,ij->i"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 8: Broadcasting (ij,j->ij)
print("=" * 60)
print("Test Case 8: Broadcasting (ij,j->ij)")
print("=" * 60)
equation = "ij,j->ij"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 9: 4D tensor transpose (abcd->dcba)
print("=" * 60)
print("Test Case 9: 4D tensor transpose (abcd->dcba)")
print("=" * 60)
equation = "abcd->dcba"
a = np.random.randn(2, 3, 4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 10: Batch outer product (bi,bj->bij)
print("=" * 60)
print("Test Case 10: Batch outer product (bi,bj->bij)")
print("=" * 60)
equation = "bi,bj->bij"
a = np.random.randn(4, 3).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 11: Bilinear form (ij,jk,kl->il)
print("=" * 60)
print("Test Case 11: Bilinear form / Chain matmul (ij,jk,kl->il)")
print("=" * 60)
equation = "ij,jk,kl->il"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)
c = np.random.randn(5, 6).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b, c)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b, c)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 12: Batch trace (bii->b)
print("=" * 60)
print("Test Case 12: Batch trace (bii->b)")
print("=" * 60)
equation = "bii->b"
a = np.random.randn(3, 4, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 13: Complex reduction (ijk,ikl->jl)
print("=" * 60)
print("Test Case 13: Complex reduction (ijk,ikl->jl)")
print("=" * 60)
equation = "ijk,ikl->jl"
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 14: Sum over specific axes (ijk->ik)
print("=" * 60)
print("Test Case 14: Sum over specific axes (ijk->ik)")
print("=" * 60)
equation = "ijk->ik"
a = np.random.randn(2, 3, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 15: Batched bilinear (bij,bjk,bkl->bil)
print("=" * 60)
print("Test Case 15: Batched bilinear (bij,bjk,bkl->bil)")
print("=" * 60)
equation = "bij,bjk,bkl->bil"
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 5).astype(np.float32)
c = np.random.randn(2, 5, 6).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b, c)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b, c)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 16: Attention-like pattern (bhqd,bhkd->bhqk)
print("=" * 60)
print("Test Case 16: Attention-like pattern (bhqd,bhkd->bhqk)")
print("=" * 60)
equation = "bhqd,bhkd->bhqk"
a = np.random.randn(2, 4, 8, 16).astype(np.float32)  # batch, heads, seq_q, dim
b = np.random.randn(2, 4, 12, 16).astype(np.float32)  # batch, heads, seq_k, dim

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 17: Complex broadcast multiplication (ij,ik,il->ijkl)
print("=" * 60)
print("Test Case 17: Complex broadcast multiplication (ij,ik,il->ijkl)")
print("=" * 60)
equation = "ij,ik,il->ijkl"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 5).astype(np.float32)
c = np.random.randn(3, 6).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b, c)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b, c)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 18: Hadamard product with sum (ij,ij->)
print("=" * 60)
print("Test Case 18: Hadamard product with sum (ij,ij->)")
print("=" * 60)
equation = "ij,ij->"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 19: Tensor contraction (ijkl,jklm->im)
print("=" * 60)
print("Test Case 19: Tensor contraction (ijkl,jklm->im)")
print("=" * 60)
equation = "ijkl,jklm->im"
a = np.random.randn(2, 3, 4, 5).astype(np.float32)
b = np.random.randn(3, 4, 5, 6).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 20: Implicit summation (ij,jk)
print("=" * 60)
print("Test Case 20: Implicit summation (ij,jk)")
print("=" * 60)
equation = "ij,jk"
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 21: Batch-wise broadcasting (LayerNorm pattern: bij,bj->bij)
print("=" * 60)
print("Test Case 21: Batch-wise broadcasting (bij,bj->bij) - LayerNorm pattern")
print("=" * 60)
equation = "bij,bj->bij"
a = np.random.randn(2, 3, 4).astype(np.float32)  # batch, seq_len, hidden_dim
b = np.random.randn(2, 4).astype(np.float32)      # batch, hidden_dim

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 22: Higher-dimensional batch-wise broadcasting (bhwc,hc->bhwc)
print("=" * 60)
print("Test Case 22: 4D batch-wise broadcasting (bhwc,hc->bhwc) - GroupNorm pattern")
print("=" * 60)
equation = "bhwc,hc->bhwc"
a = np.random.randn(2, 3, 4, 5).astype(np.float32)  # batch, height, width, channels
b = np.random.randn(3, 5).astype(np.float32)        # height, channels

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 23: Singleton dimension handling (ijk->ik where j=1)
print("=" * 60)
print("Test Case 23: Singleton dimension removal (ijk->ik with j=1) - ONNX unsqueeze artifact")
print("=" * 60)
equation = "ijk->ik"
a = np.random.randn(3, 1, 4).astype(np.float32)  # i=3, singleton=1, k=4

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 24: Singleton dimension with broadcasting (bij,blj->bij where l=1)
print("=" * 60)
print("Test Case 24: Singleton dimension with broadcasting (bij,blj->bij where l=1)")
print("=" * 60)
equation = "bij,blj->bij"
a = np.random.randn(2, 3, 4).astype(np.float32)     # batch, i, j
b = np.random.randn(2, 1, 4).astype(np.float32)     # batch, singleton, j

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 25: Non-contiguous repeated indices (ibi->b) - GNN adjacency pattern
print("=" * 60)
print("Test Case 25: Non-contiguous repeated indices (ibi->b) - GNN pattern")
print("=" * 60)
equation = "ibi->b"
# Create a 3D tensor where first and last dimensions are the same
dim_size = 5
a = np.random.randn(dim_size, 3, dim_size).astype(np.float32)

t0 = time.time()
expected = onnx_einsum(equation, a)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

# Test Case 26: Multiple independent reductions (ij,kl->) - Combined loss pattern
print("=" * 60)
print("Test Case 26: Multiple independent reductions (ij,kl->) - Combined loss pattern")
print("=" * 60)
equation = "ij,kl->"
a = np.random.randn(3, 4).astype(np.float32)  # No shared dimensions
b = np.random.randn(5, 6).astype(np.float32)  # No shared dimensions

t0 = time.time()
expected = onnx_einsum(equation, a, b)
print(f"NumPy: {expected.shape}, {time.time() - t0:.4f} seconds")

t0 = time.time()
einsum_op = EinsumOp(mgr, equation)
result = einsum_op.run(a, b)[0]
print(f"EinsumOp: {result.shape}, {time.time() - t0:.4f} seconds")

print(f"Shape match: {result.shape == expected.shape}")
print(f"Max error: {np.abs(result - expected).max():.6e}")
print(f"All close: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
print()

