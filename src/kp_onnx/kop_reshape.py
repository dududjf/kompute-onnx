import numpy as np
import kp


class ReshapeOp:
    def __init__(self, manager: kp.Manager, allowzero: int = 0):
        self.manager = manager
        self.allowzero = int(allowzero)

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        data = inputs[0]
        shape = np.array(inputs[1], dtype=int)

        if len(inputs) >= 3 and inputs[2] is not None:
            allowzero = int(inputs[2]) == 1
        else:
            allowzero = int(getattr(self, "allowzero", 0)) == 1

        new_shape = np.copy(shape)
        if not allowzero:
            zeros_idx = np.where(new_shape == 0)
            new_shape[zeros_idx] = np.array(data.shape)[zeros_idx]

        neg_idx = None
        known_prod = 1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_prod *= dim
        total = int(np.prod(data.shape))
        if neg_idx is not None:
            if known_prod == 0 or total % known_prod != 0:
                raise ValueError("Cannot infer -1 dimension")
            new_shape[neg_idx] = total // known_prod

        if data.size == 0:
            return [data.reshape(new_shape)]

        input_tensor = self.manager.tensor(data.reshape(-1).astype(np.float32))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse([(input_tensor, list(new_shape))],updated_algorithms,updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        output = data.reshape(new_shape)

        del input_tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, tensor_shape = input_tensors[0]
        prod = int(np.prod(tensor_shape))
        tensor_out = self.manager.tensor(np.zeros(prod, dtype=np.float32))
        updated_tensors.append(tensor_out)
        return [(tensor_out, tensor_shape)]
