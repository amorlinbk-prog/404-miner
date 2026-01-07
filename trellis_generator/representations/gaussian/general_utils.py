import torch


def inverse_sigmoid(x: torch.tensor) -> torch.tensor:
    return torch.log(x / (1.0 - x))


def strip_lowerdiag(L: torch.tensor) -> torch.tensor:
    # Define the indices for the lower triangular part, including the diagonal
    tril_indices = torch.tril_indices(row=3, col=3, offset=0, device=L.device)

    # Extract the lower triangular elements from each matrix in the batch
    lower_triangular_elements = L[:, tril_indices[0], tril_indices[1]]

    # Select the specific elements corresponding to the desired positions
    uncertainty = lower_triangular_elements[:, [0, 1, 2, 4, 5, 8]]

    return uncertainty


def strip_symmetric(sym: torch.tensor) -> torch.tensor:
    return strip_lowerdiag(sym)


def build_rotation_matrices(quaternions: torch.tensor) -> torch.tensor:
    # Normalize the quaternions
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=1)

    # Extract individual components
    r, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the rotation matrices
    rotation_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)

    return rotation_matrices


def build_scaled_rotation_matrices(scaling_mat: torch.tensor, quaternions: torch.tensor) -> torch.tensor:
    batch_size = scaling_mat.shape[0]

    # Initialize scaling matrices L as identity matrices
    L = torch.eye(3, device=scaling_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Set the diagonal elements to the scaling factors
    L *= scaling_mat.unsqueeze(2)

    # Compute rotation matrices R using the provided build_rotation function
    R = build_rotation_matrices(quaternions)

    # Perform batched matrix multiplication
    L = torch.bmm(R, L)

    return L
