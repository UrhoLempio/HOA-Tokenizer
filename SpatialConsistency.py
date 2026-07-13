#!/usr/bin/env python3
"""
Bare minimum Spatial Consistency class for DirAC-based spatial audio evaluation.

This class computes frame-by-frame spatial consistency between reference and target
First Order Ambisonics signals in Ambix format, using cosine similarity of intensity
vectors weighted by energy and directional strength (1 - diffuseness).
"""

import torch


def _to_torch_audio(audio_signal):
    tensor = audio_signal
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("SpatialConsistency expects torch.Tensor inputs")

    if tensor.dim() == 2:
        if tensor.shape[-1] != 4:
            tensor = tensor.transpose(0, 1)
        if tensor.shape[-1] != 4:
            raise ValueError("Expected audio signal with shape (n_samples, 4)")
        return tensor.to(dtype=torch.float32, device=tensor.device).unsqueeze(0)

    if tensor.dim() == 3:
        if tensor.shape[-1] != 4:
            if tensor.shape[1] == 4:
                tensor = tensor.transpose(1, 2)
            else:
                raise ValueError("Expected audio signal with shape (batch, n_samples, 4)")
        return tensor.to(dtype=torch.float32, device=tensor.device)

    raise ValueError("Expected audio signal with shape (n_samples, 4) or (batch, n_samples, 4)")


def _maybe_to_python_scalar(value):
    return value.detach().item() if value.numel() == 1 else value.detach()


def vertical_to_interaural_deg(azimuth_deg, elevation_deg):
    az = torch.deg2rad(azimuth_deg)
    el = torch.deg2rad(elevation_deg)
    x = torch.cos(el) * torch.cos(az)
    y = torch.cos(el) * torch.sin(az)
    z = torch.sin(el)
    lateral = torch.asin(y)
    polar = torch.atan2(z, x)
    return torch.rad2deg(lateral), torch.rad2deg(polar)


def interaural_to_vertical_deg(lateral_deg, polar_deg):
    """
    Convert interaural spherical coordinates (lateral, polar) to vertical coordinates
    (azimuth, elevation) in degrees.
    """
    lateral = torch.deg2rad(lateral_deg)
    polar = torch.deg2rad(polar_deg)
    x = torch.cos(polar) * torch.cos(lateral)
    y = torch.sin(lateral)
    z = torch.sin(polar) * torch.cos(lateral)
    azimuth = torch.rad2deg(torch.atan2(y, x))
    elevation = torch.rad2deg(torch.asin(z / torch.sqrt(x**2 + y**2 + z**2)))
    return azimuth, elevation


def vertical_to_interaural(azimuth, elevation):
    lateral, polar = vertical_to_interaural_deg(torch.rad2deg(azimuth), torch.rad2deg(elevation))
    return torch.deg2rad(lateral), torch.deg2rad(polar)


def interaural_to_vertical(lateral, polar):
    """
    Convert interaural spherical coordinates (lateral, polar) to vertical coordinates
    (azimuth, elevation) in radians.
    """
    azimuth, elevation = interaural_to_vertical_deg(torch.rad2deg(lateral), torch.rad2deg(polar))
    return torch.deg2rad(azimuth), torch.deg2rad(elevation)


def cartesian_to_interaural(x, y, z):
    """
    Convert Cartesian (x: forward, y: left, z: up) to spherical (lateral, polar, distance)
    in interaural coordinates.
    Returns angles in radians.
    """
    r = torch.sqrt(x**2 + y**2 + z**2)
    polar = torch.atan2(z, x)
    lateral = torch.zeros_like(r)
    valid_mask = r > torch.finfo(r.dtype).eps
    lateral = torch.where(valid_mask, torch.asin(y / r), lateral)
    return lateral, polar, r


def interaural_to_cartesian(lateral, polar, r):
    """
    Convert spherical (lateral, polar, distance) to Cartesian (x: forward, y: left, z: up)
    in interaural coordinates.
    Angles should be in radians.
    """
    x = r * torch.cos(polar) * torch.cos(lateral)
    z = r * torch.sin(polar) * torch.cos(lateral)
    y = r * torch.sin(lateral)
    return x, y, z

class SpatialConsistency:
    """
    Minimal spatial consistency calculator for FOA Ambix signals.
    
    Computes weighted cosine similarity between intensity vectors of reference
    and target audio signals. Uses reference signal for weighting calculation.
    """
    
    def __init__(self, 
                 sample_rate: int = 24000,
                 window_size: int = 1024,
                 hop_length: int = 256,
                 energy_weighting: bool = False,
                 diffuseness_weighting: bool = True,
                 energy_threshold: float = 1e-6,
                 diffuseness_threshold: float = 0.5):
        """
        Initialize the spatial consistency calculator.
        
        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate in Hz
        window_size : int
            STFT window size in samples
        hop_length : int
            STFT hop length in samples
        energy_weighting : bool
            Whether to apply energy weighting
        diffuseness_weighting : bool
            Whether to apply diffuseness weighting
        energy_threshold : float
            Minimum energy threshold for valid time-frequency points
        diffuseness_threshold : float
            Maximum diffuseness for valid time-frequency points (0-1)
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.energy_weighting = energy_weighting
        self.diffuseness_weighting = diffuseness_weighting
        self.energy_threshold = energy_threshold
        self.diffuseness_threshold = diffuseness_threshold
    
    def _stft(self, audio_signal):
        """
        Perform STFT on Ambix format audio signal.
        
        Parameters:
        -----------
        audio_signal : ndarray or torch.Tensor
            Input audio signal of shape (n_samples, 4) or (batch, n_samples, 4) in Ambix format [W, Y, Z, X]
        
        Returns:
        --------
        stft_frames : torch.Tensor
            STFT frames of shape (batch, n_frames, n_freq_bins, 4) containing complex values
        """
        waveform = _to_torch_audio(audio_signal)
        if waveform.dim() != 3:
            raise ValueError("Expected audio signal with shape (n_samples, 4) or (batch, n_samples, 4)")

        batch_size, n_samples, n_channels = waveform.shape
        if n_channels != 4:
            raise ValueError("SpatialConsistency expects 4-channel FOA input")

        waveform = waveform.transpose(1, 2)  # Shape: (batch, 4, n_samples)
        waveform_flat = waveform.reshape(batch_size * 4, n_samples)

        window = torch.hann_window(
            self.window_size,
            device=waveform_flat.device,
            dtype=waveform_flat.dtype,
        )

        stft_result = torch.stft(
            waveform_flat,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            return_complex=True,
        )

        freq_bins = stft_result.size(1)
        n_frames = stft_result.size(2)
        stft_result = stft_result.view(batch_size, 4, freq_bins, n_frames)
        return stft_result.permute(0, 3, 2, 1)
    
    def _extract_dirac_parameters(self, stft_frames):
        """
        Extract DirAC parameters directly from Ambix STFT frames.
        
        Parameters:
        -----------
        stft_frames : torch.Tensor
            STFT frames of shape (batch, n_frames, n_freq_bins, 4) [W, Y, Z, X]
        
        Returns:
        --------
        intensity_vectors : torch.Tensor
            Intensity vectors of shape (batch, n_frames, n_freq_bins, 3) [X, Y, Z components]
        diffuseness : torch.Tensor
            Diffuseness values of shape (batch, n_frames, n_freq_bins)
        energy : torch.Tensor
            Energy values of shape (batch, n_frames, n_freq_bins)
        """
        if stft_frames.dim() == 3:
            stft_frames = stft_frames.unsqueeze(0)

        W = stft_frames[..., 0]
        Y = stft_frames[..., 1]
        Z = stft_frames[..., 2]
        X = stft_frames[..., 3]

        XYZ = torch.stack([X, Y, Z], dim=-1)
        #XYZ = XYZ * torch.sqrt(torch.tensor(3.0, device=XYZ.device, dtype=XYZ.dtype))
        XYZ = XYZ * (3.0 ** 0.5)

        intensity_vectors = torch.real(torch.conj(W).unsqueeze(-1) * XYZ)
        energy = torch.abs(W)**2 + torch.sum(torch.abs(XYZ)**2, dim=-1) / 2.0

        intensity_magnitude = torch.sqrt(torch.sum(intensity_vectors**2, dim=-1))
        eps = torch.finfo(XYZ.dtype).eps
        diffuseness = 1.0 - intensity_magnitude / (energy + eps)
        diffuseness = torch.clamp(diffuseness, eps, 1.0 - eps)

        return intensity_vectors, diffuseness, energy
    
    def compute_spatial_consistency(self, 
                                  reference_audio: torch.Tensor, 
                                  target_audio: torch.Tensor,
                                  lateral_only: bool = False):
        """
        Compute spatial consistency between reference and target audio.
        
        Parameters:
        -----------
        reference_audio : ndarray or torch.Tensor
            Reference audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
        target_audio : ndarray or torch.Tensor
            Target audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
        lateral_only : bool
            If True, compute loss only in terms of lateral angle (ignore elevation and front-back confusions)
            
        Returns:
        --------
        loss : torch.Tensor or float
            Spatial consistency loss value (lower is better)
        """
        ref_stft = self._stft(reference_audio)
        target_stft = self._stft(target_audio)

        ref_intensity, ref_diffuseness, ref_energy = self._extract_dirac_parameters(ref_stft)
        target_intensity, _, _ = self._extract_dirac_parameters(target_stft)

        ref_norm = torch.linalg.vector_norm(ref_intensity, dim=-1)
        target_norm = torch.linalg.vector_norm(target_intensity, dim=-1)
        if lateral_only:
            ref_lateral, ref_polar, ref_distance = cartesian_to_interaural(
                ref_intensity[..., 0], ref_intensity[..., 1], ref_intensity[..., 2])
            target_lateral, target_polar, target_distance = cartesian_to_interaural(
                target_intensity[..., 0], target_intensity[..., 1], target_intensity[..., 2])
            ref_intensity_x, ref_intensity_y, ref_intensity_z = interaural_to_cartesian(ref_lateral, 0, ref_distance)
            target_intensity_x, target_intensity_y, target_intensity_z = interaural_to_cartesian(target_lateral, 0, target_distance)
            ref_intensity = torch.stack([ref_intensity_x, ref_intensity_y, ref_intensity_z], dim=-1)
            target_intensity = torch.stack([target_intensity_x, target_intensity_y, target_intensity_z], dim=-1)

        dot_products = torch.sum(ref_intensity * target_intensity, dim=-1)

        eps = torch.finfo(ref_norm.dtype).eps
        if self.energy_threshold is not None:
            energy_mask = ref_energy > self.energy_threshold
            direction_mask_ref = ref_norm > self.energy_threshold
        else:
            energy_mask = torch.ones(ref_energy.shape, dtype=torch.bool, device=ref_energy.device)
            direction_mask_ref = torch.ones(ref_norm.shape, dtype=torch.bool, device=ref_norm.device)

        if self.diffuseness_threshold is not None:
            diffuse_mask = ref_diffuseness < self.diffuseness_threshold
        else:
            diffuse_mask = torch.ones(ref_diffuseness.shape, dtype=torch.bool, device=ref_diffuseness.device)

        valid_mask = energy_mask & direction_mask_ref & diffuse_mask
        mask_ratio = valid_mask.float().mean()
        if not torch.any(valid_mask):
            zero = torch.zeros((), device=ref_energy.device, dtype=ref_energy.dtype)
            return zero, torch.zeros_like(zero)

        cosine_sim = torch.zeros_like(dot_products)
        denom = torch.clamp(ref_norm * target_norm, min=eps)
        cosine_sim = torch.where(valid_mask, dot_products / denom, cosine_sim)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        weights = torch.ones_like(ref_energy)
        if self.energy_weighting:
            weights *= ref_energy
        if self.diffuseness_weighting:
            weights *= (1.0 - ref_diffuseness)

        weights = weights * valid_mask.to(weights.dtype)
        cosine_loss = 1.0 - cosine_sim
        loss = torch.sum(cosine_loss * weights) / (torch.sum(weights) + eps)

        return loss, mask_ratio

    def estimate_direction_of_arrival(self, audio_signal: torch.Tensor):
        """
        Estimate direction of arrival from a single audio signal.
        
        Parameters:
        -----------
        audio_signal : ndarray or torch.Tensor
            Input audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
            
        Returns:
        --------
        azimuth : torch.Tensor or float
            Azimuth angle in degrees (-180 to 180, where 0° is front, 90° is left)
        elevation : torch.Tensor or float
            Elevation angle in degrees (-90 to 90, where 0° is horizontal, 90° is up)
        """
        stft_frames = self._stft(audio_signal)

        rand_azi = torch.tensor(torch.pi, dtype=torch.float32, device=audio_signal.device)
        rand_ele = torch.tensor(-90.0, dtype=torch.float32, device=audio_signal.device)

        intensity_vectors, diffuseness, energy = self._extract_dirac_parameters(stft_frames)

        if self.energy_threshold is not None:
            energy_mask = energy > self.energy_threshold
        else:
            energy_mask = torch.ones(energy.shape, dtype=torch.bool, device=energy.device)
        if self.diffuseness_threshold is not None:
            diffuseness_mask = diffuseness < self.diffuseness_threshold
        else:
            diffuseness_mask = torch.ones(diffuseness.shape, dtype=torch.bool, device=diffuseness.device)
        valid_mask = energy_mask & diffuseness_mask

        if not torch.any(valid_mask):
            return _maybe_to_python_scalar(rand_azi), _maybe_to_python_scalar(rand_ele), None

        weights = torch.ones_like(energy)
        if self.energy_weighting:
            weights *= energy
        if self.diffuseness_weighting:
            weights *= (1.0 - diffuseness)
        weights = weights * valid_mask.to(weights.dtype)
        total_weight = torch.sum(weights)
        if total_weight == 0:
            return _maybe_to_python_scalar(rand_azi), _maybe_to_python_scalar(rand_ele), None

        avg_intensity = torch.sum(intensity_vectors * weights[..., None], dim=(0, 1, 2)) / total_weight
        squared_diff = (intensity_vectors - avg_intensity[None, None, :])**2
        weighted_squared_diff = squared_diff * weights[..., None]
        variance = torch.sum(weighted_squared_diff, dim=(0, 1, 2)) / total_weight
        avg_intensity_variance = torch.sum(variance)

        x, y, z = avg_intensity[0], avg_intensity[1], avg_intensity[2]
        azimuth_rad = torch.atan2(y, x)
        azimuth_deg = torch.rad2deg(azimuth_rad)
        r_xy = torch.sqrt(x**2 + y**2)
        elevation_rad = torch.atan2(z, r_xy)
        elevation_deg = torch.rad2deg(elevation_rad)

        return azimuth_deg, elevation_deg, avg_intensity_variance

def compute_spatial_consistency_loss(reference_audio: torch.Tensor, 
                                   target_audio: torch.Tensor,
                                   sample_rate: int = 24000,
                                   window_size: int = 1024,
                                   hop_length: int = 256,
                                   lateral_only: bool = False) -> float:
    """
    Convenience function to compute spatial consistency loss.
    
    Parameters:
    -----------
    reference_audio : ndarray
        Reference audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
    target_audio : ndarray
        Target audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
    sample_rate : int
        Audio sampling rate
    window_size : int
        STFT window size
    hop_length : int
        STFT hop length
    lateral_only : bool
        If True, compute loss only in terms of lateral angle (ignore elevation and front-back confusions)
        
    Returns:
    --------
    loss : float
        Spatial consistency loss value
    """
    calculator = SpatialConsistency(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_length=hop_length
    )
    return calculator.compute_spatial_consistency(reference_audio, target_audio, lateral_only=lateral_only)


def estimate_direction_of_arrival(audio_signal: torch.Tensor,
                                 sample_rate: int = 24000,
                                 window_size: int = 1024,
                                 hop_length: int = 256,
                                 energy_weighting: bool = True,
                                 diffuseness_weighting: bool = True,
                                 energy_threshold: float = 1e-6,
                                 diffuseness_threshold: float = 0.95):
    """
    Convenience function to estimate direction of arrival from audio signal.
    
    Parameters:
    -----------
    audio_signal : ndarray
        Input audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
    sample_rate : int
        Audio sampling rate
    window_size : int
        STFT window size
    hop_length : int
        STFT hop length
    energy_weighting : bool
        Whether to apply energy weighting
    diffuseness_weighting : bool
        Whether to apply diffuseness weighting
    energy_threshold : float
        Minimum energy threshold for valid time-frequency points
    diffuseness_threshold : float
        Maximum diffuseness for valid time-frequency points (0-1)
        
    Returns:
    --------
    azimuth : float
        Azimuth angle in degrees (-180 to 180, where 0° is front, 90° is left)
    elevation : float
        Elevation angle in degrees (-90 to 90, where 0° is horizontal, 90° is up)
    """
    calculator = SpatialConsistency(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_length=hop_length,
        energy_weighting=energy_weighting,
        diffuseness_weighting=diffuseness_weighting,
        energy_threshold=energy_threshold,
        diffuseness_threshold=diffuseness_threshold
    )
    return calculator.estimate_direction_of_arrival(
        audio_signal,
    )


if __name__ == "__main__":
    torch.manual_seed(1234)
    n_samples = 8192
    reference = torch.randn(n_samples, 4)
    target = reference + 0.01 * torch.randn(n_samples, 4)

    calculator = SpatialConsistency()
    loss, mask_ratio = calculator.compute_spatial_consistency(
    reference,
    target,
)

    print(f"loss={loss.item():.4f}")
    print(f"mask_ratio={mask_ratio.item():.4f}")

    
