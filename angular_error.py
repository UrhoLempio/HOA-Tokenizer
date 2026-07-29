import torch
from SpatialConsistency import estimate_direction_of_arrival

def angular_error(az1, el1, az2, el2, eps=1e-8):
    """
    Angular error between two vecs.

    Args:
        az1, el1: Predicted azimuth and elevation (radians)
        az2, el2: Ground truth azimuth and elevation (radians)
            Can be scalars or tensors of the same shape.

    Returns:
        Angular error in radians.
    """

    #  spherical to Cartesian unit vectors
    x1 = torch.cos(el1) * torch.cos(az1)
    y1 = torch.cos(el1) * torch.sin(az1)
    z1 = torch.sin(el1)

    x2 = torch.cos(el2) * torch.cos(az2)
    y2 = torch.cos(el2) * torch.sin(az2)
    z2 = torch.sin(el2)

    # dot product
    dot = x1 * x2 + y1 * y2 + z1 * z2

    # clamp dot product to avoid numerical issues 
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    # Angular error in radians
    return torch.acos(dot)

if __name__ == "__main__":
    # Test the angular_error function with some example values
    in_channels = 4
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    audio_input = torch.randn(batch_size, in_channels, 24000).to(device)
    audio_hat = torch.randn(batch_size, in_channels, 24000).to(device)

    az1, el1, _ = estimate_direction_of_arrival(audio_hat)
    print(f"Predicted azimuth: {az1}, Predicted elevation: {el1}")
    az2, el2, _ = estimate_direction_of_arrival(audio_input)
    print(f"Ground truth azimuth: {az2}, Ground truth elevation: {el2}")
    error = angular_error(az1, el1, az2, el2)
    print("Angular error (radians):", error)