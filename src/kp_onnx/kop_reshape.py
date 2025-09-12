import numpy as np
import kp


class ReshapeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 2, "ReshapeOp requires at least data and target_shape"
        data = inputs[0]
        target_shape = np.array(inputs[1], dtype=int)
        allowzero = int(inputs[2]) if len(inputs) >= 3 and inputs[2] is not None else 0

        tensor_data = self.manager.tensor(data.reshape(-1).astype(np.float32))
        tensor_shape = self.manager.tensor(target_shape.astype(np.int32))
        tensor_allowzero = self.manager.tensor(np.array([allowzero], dtype=np.int32))

        input_tensors = [
            (tensor_data, list(data.shape)),
            (tensor_shape, list(target_shape.shape)),
            (tensor_allowzero, [])
        ]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        tensor_out.data()[:] = tensor_data.data()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        tensor_data, input_shape_list = input_tensors[0]
        tensor_shape, _ = input_tensors[1]
        tensor_allowzero, _ = input_tensors[2]

        target_shape_list = tensor_shape.data().astype(int).tolist()
        allowzero = bool(tensor_allowzero.data()[0])

        new_shape = [0] * len(target_shape_list)
        for i, dim in enumerate(target_shape_list):
            if dim != 0 or allowzero:
                new_shape[i] = dim
            else:
                new_shape[i] = input_shape_list[i]

        neg_idx = -1
        known_prod = 1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                neg_idx = i
            else:
                known_prod *= dim
        total = np.prod(input_shape_list)
        if neg_idx != -1:
            new_shape[neg_idx] = total // known_prod

        tensor_out = self.manager.tensor(np.zeros(np.prod(new_shape), dtype=np.float32))
        updated_tensors.append(tensor_out)

        return [(tensor_out, new_shape)]
