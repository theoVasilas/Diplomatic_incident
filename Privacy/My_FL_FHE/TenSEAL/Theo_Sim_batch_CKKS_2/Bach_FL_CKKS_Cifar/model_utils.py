import torch
import numpy as np
from collections import OrderedDict
from typing import List, Tuple


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # replace the parameters
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def flatten_params(params: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Flatten model parameters into a single NumPy array."""
    shapes = [p.shape for p in params]  # Store original shapes
    flattened = np.concatenate([p.flatten() for p in params])  # Flatten and concatenate
    return flattened, shapes

def reshape_params(flattened: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    """Reshape a flattened NumPy array back into original parameter shapes."""
    params = []
    index = 0
    for shape in shapes:
        size = np.prod(shape)  # Number of elements in this parameter
        params.append(flattened[index : index + size].reshape(shape))  # Reshape
        index += size
    return params

def test_flatten_reshape(original_params, new_params):
    sum = 0
    for orig, new, shape in zip(original_params, new_params, [p.shape for p in original_params]):
        # print(f"Layer shape: {shape}")
        
        # Compute error (Euclidean norm of the difference)
        error = np.linalg.norm(orig - new)
        # print(f"  Error: {error:.6f}")
        # Show sample values from both
        # print(f"  Original values: {orig.flatten()[:5]}")  # Show first 5 values
        # print(f"  New values: {new.flatten()[:5]}\n")  # Show first 5 values

        sum += error
    print(f"Total error: {sum:.6f}")

def batch_parameters(params: List[np.ndarray], slot_count: int) -> List[List[np.ndarray]]:
    batched = [params[i:i + slot_count] for i in range(0, len(params), slot_count)]
    # print(f"Number of parts: {len(batched)}")
    return batched

