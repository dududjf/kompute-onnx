from kp import Manager
import numpy as np
import time
from kp_onnx.kop_sequence_insert import SequenceInsertOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

seq_insert_op = SequenceInsertOp(mgr)


def onnx_sequence_insert_reference(sequence, tensor, position=None):
    """ONNX reference implementation of SequenceInsert"""
    # Make a copy of input sequence
    seq = []
    if sequence is not None and (
        not isinstance(sequence, np.ndarray) or len(sequence.shape) > 0
    ):
        seq.extend(sequence)

    if position is not None:
        # Handle position as array or scalar
        if isinstance(position, np.ndarray):
            if position.size == 1:
                pos = int(position.flat[0])
            else:
                pos = int(position[0])
        else:
            pos = int(position)
        insert_position = (pos + len(seq)) % max(len(seq), 1)
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence
        seq.append(tensor)
    return seq


# Case 1: Insert tensor into middle of sequence with explicit position
print("\nCase 1: Insert tensor into middle of sequence at position 1")

# Create initial sequence with 3 tensors
x1 = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
x2 = np.random.uniform(-1, 1, (4, 5)).astype(np.float32)
x3 = np.random.uniform(-1, 1, (3,)).astype(np.float32)
sequence = [x1, x2, x3]

# Tensor to insert
tensor_to_insert = np.random.uniform(-1, 1, (6, 2)).astype(np.float32)

# Position to insert at
position = np.array([1], dtype=np.int64)

print(f"Initial sequence length: {len(sequence)}")
print(f"Initial sequence shapes: {[x.shape for x in sequence]}")
print(f"Tensor to insert shape: {tensor_to_insert.shape}")
print(f"Insert position: {position[0]}")

# NumPy reference
t0 = time.time()
numpy_out = onnx_sequence_insert_reference(sequence, tensor_to_insert, position)
print(f"NumPy: {time.time() - t0} seconds")

# Kompute implementation
t1 = time.time()
kp_out = seq_insert_op.run(sequence, tensor_to_insert, position)
print(f"{seq_insert_op}: {time.time() - t1} seconds")

# Verify results
print(f"\nOutput sequence length: NumPy={len(numpy_out)}, Kompute={len(kp_out)}")
print(f"Expected length: {len(sequence) + 1}")

print("\nOutput sequence shapes:")
for i in range(len(numpy_out)):
    print(f"  Position {i}: NumPy={numpy_out[i].shape}, Kompute={kp_out[i].shape}")

# Check if all tensors match
all_close = True
for i, (numpy_tensor, kp_tensor) in enumerate(zip(numpy_out, kp_out)):
    is_close = np.allclose(numpy_tensor, kp_tensor, rtol=1e-4, atol=1e-4)
    print(f"Tensor {i} all close: {is_close}")
    all_close = all_close and is_close

print(f"\nAll tensors match: {all_close}")

# Verify insertion order
print(f"\nVerify insertion: Tensor at position 1 should be the inserted tensor")
print(f"Match with inserted tensor: {np.allclose(kp_out[1], tensor_to_insert, rtol=1e-4, atol=1e-4)}")

