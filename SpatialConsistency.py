#!/usr/bin/env python3
"""
Bare minimum Spatial Consistency class for DirAC-based spatial audio evaluation.

This class computes frame-by-frame spatial consistency between reference and target
First Order Ambisonics signals in Ambix format, using cosine similarity of intensity
vectors weighted by energy and directional strength (1 - diffuseness).
"""

import numpy as np
import torch

import numpy as np

def vertical_to_interaural_deg(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Convert to Cartesian
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    # Convert to interaural spherical coordinates
    lateral = np.arcsin(y)  # lateral angle
    polar = np.arctan2(z, x)  # polar angle
    return np.degrees(lateral), np.degrees(polar)

def interaural_to_vertical_deg(lateral_deg, polar_deg):
    """
    Convert interaural spherical coordinates (lateral, polar) to vertical coordinates
    (azimuth, elevation) in degrees.
    """
    lateral = np.radians(lateral_deg)
    polar = np.radians(polar_deg)

    # Convert to Cartesian
    # lateral is angle off median plane (x-z), so y = sin(lateral)
    # polar is rotation about y-axis, affecting x and z components
    x = np.cos(polar) * np.cos(lateral)
    y = np.sin(lateral)
    z = np.sin(polar) * np.cos(lateral)

    # Convert back to azimuth and elevation
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    
    return azimuth, elevation

def vertical_to_interaural(azimuth, elevation):
    lateral, polar = vertical_to_interaural_deg(np.degrees(azimuth), np.degrees(elevation))
    return np.radians(lateral), np.radians(polar)

def interaural_to_vertical(lateral, polar):
    """
    Convert interaural spherical coordinates (lateral, polar) to vertical coordinates
    (azimuth, elevation) in radians.
    """
    azimuth, elevation = interaural_to_vertical_deg(np.degrees(lateral),  np.degrees(polar))
    return np.radians(azimuth), np.radians(elevation)

def cartesian_to_interaural(x, y, z):
    """
    Convert Cartesian (x: forward, y: left, z: up) to spherical (lateral, polar, distance)
    in interaural coordinates.
    Returns angles in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arctan2(z, x)  # polar angle: rotation about y-axis (front-back)
    lateral = np.zeros_like(r)
    valid_mask = r > np.finfo(float).eps
    lateral[valid_mask] = np.arcsin(y[valid_mask] / r[valid_mask])  # lateral angle: angle off median plane (left-right)
    return lateral, polar, r

def interaural_to_cartesian(lateral, polar, r):
    """
    Convert spherical (lateral, polar, distance) to Cartesian (x: forward, y: left, z: up)
    in interaural coordinates.
    Angles should be in radians.
    """
    x = r * np.cos(polar) * np.cos(lateral)
    z = r * np.sin(polar) * np.cos(lateral)
    y = r * np.sin(lateral)
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
        audio_signal : ndarray
            Input audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
        
        Returns:
        --------
        stft_frames : ndarray
            STFT frames of shape (n_frames, n_freq_bins, 4) containing complex values
        """
        # Convert to torch tensor and transpose for torchaudio (channels first)
        waveform = torch.tensor(audio_signal.T).float()  # Shape: (4, n_samples)
        
        # Compute STFT for all channels
        stft_result = torch.stft(
            waveform,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size),
            center=True,
            pad_mode='reflect',
            normalized=False,
            return_complex=True
        )
        
        # stft_result shape: (4, n_freq_bins, n_frames)
        # Transpose to (n_frames, n_freq_bins, 4)
        stft_frames = stft_result.permute(2, 1, 0).numpy()
        
        return stft_frames
    
    def _extract_dirac_parameters(self, stft_frames):
        """
        Extract DirAC parameters directly from Ambix STFT frames.
        
        Parameters:
        -----------
        stft_frames : ndarray
            STFT frames of shape (n_frames, n_freq_bins, 4) [W, Y, Z, X]
        
        Returns:
        --------
        intensity_vectors : ndarray
            Intensity vectors of shape (n_frames, n_freq_bins, 3) [X, Y, Z components]
        diffuseness : ndarray
            Diffuseness values of shape (n_frames, n_freq_bins)
        energy : ndarray
            Energy values of shape (n_frames, n_freq_bins)
        """
        n_frames, n_freq_bins, _ = stft_frames.shape
        
        # Extract W (omnidirectional) and YZX (directional) components in Ambix ACN order
        W = stft_frames[:, :, 0]  # Shape: (n_frames, n_freq_bins)
        Y = stft_frames[:, :, 1]  # Shape: (n_frames, n_freq_bins)
        Z = stft_frames[:, :, 2]  # Shape: (n_frames, n_freq_bins)
        X = stft_frames[:, :, 3]  # Shape: (n_frames, n_freq_bins)
        
        # Rearrange to XYZ order for intensity vector calculation
        XYZ = np.stack([X, Y, Z], axis=-1)  # Shape: (n_frames, n_freq_bins, 3)
        
        # undo SN3D normalization
        XYZ = XYZ * np.sqrt(3)

        # Compute intensity = Re(conj(W) * [X, Y, Z])
        intensity_vectors = np.real(np.conj(W)[:, :, np.newaxis] * XYZ)
        
        # Compute energy density = (abs(w).^2 + sum(abs(X).^2,2))/2 --> formula in paper: (abs(w)^2 + sum(abs(XYZ)^2 / 2))
        energy = (np.abs(W)**2 + (np.sum(np.abs(XYZ)**2, axis=2) / 2)) # mistake in original code?

        # Compute diffuseness
        intensity_magnitude = np.sqrt(np.sum(intensity_vectors**2, axis=2))
        diffuseness = 1 - intensity_magnitude / (energy + np.finfo(float).eps)
        diffuseness = np.clip(diffuseness, np.finfo(float).eps, 1 - np.finfo(float).eps)
        
        return intensity_vectors, diffuseness, energy
    
    def compute_spatial_consistency(self, 
                                  reference_audio: np.ndarray, 
                                  target_audio: np.ndarray,
                                  lateral_only: bool = False) -> float:
        """
        Compute spatial consistency between reference and target audio.
        
        Parameters:
        -----------
        reference_audio : ndarray
            Reference audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
        target_audio : ndarray
            Target audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
        lateral_only : bool
            If True, compute loss only in terms of lateral angle (ignore elevation and front-back confusions)
            
        Returns:
        --------
        loss : float
            Spatial consistency loss value (lower is better)
        """
        # Extract STFT frames
        ref_stft = self._stft(reference_audio)
        target_stft = self._stft(target_audio)
        
        # Extract DirAC parameters
        ref_intensity, ref_diffuseness, ref_energy = self._extract_dirac_parameters(ref_stft)
        target_intensity, target_diffuseness, target_energy = self._extract_dirac_parameters(target_stft)

        # Compute norms for intensity vectors
        ref_norm = np.linalg.norm(ref_intensity, axis=-1)
        target_norm = np.linalg.norm(target_intensity, axis=-1)
        if lateral_only:
            # to compare lateral angles, convert intensity vectors to lat/pol/dist, set pol to 0, convert back to x/y/z
            ref_lateral, ref_polar, ref_distance = cartesian_to_interaural(
                ref_intensity[..., 0], ref_intensity[..., 1], ref_intensity[..., 2])
            target_lateral, target_polar, target_distance = cartesian_to_interaural(
                target_intensity[..., 0], target_intensity[..., 1], target_intensity[..., 2])
            ref_intensity_x, ref_intensity_y, ref_intensity_z = interaural_to_cartesian(ref_lateral, 0, ref_distance)
            target_intensity_x, target_intensity_y, target_intensity_z = interaural_to_cartesian(target_lateral, 0, target_distance)
            ref_intensity = np.stack([ref_intensity_x, ref_intensity_y, ref_intensity_z], axis=-1)
            target_intensity = np.stack([target_intensity_x, target_intensity_y, target_intensity_z], axis=-1)

        dot_products = np.sum(ref_intensity * target_intensity, axis=-1)
        
        # Create validity mask based on reference signal (as specified)
        # Optional masking based on thresholds
        if self.energy_threshold is not None:
            energy_mask = ref_energy > self.energy_threshold
            direction_mask_ref = ref_norm > self.energy_threshold
        else:
            energy_mask = np.ones(ref_energy.shape, dtype=bool)
            direction_mask_ref = np.ones(ref_norm.shape, dtype=bool)
        
        if self.diffuseness_threshold is not None:
            diffuse_mask = ref_diffuseness < self.diffuseness_threshold
        else:
            diffuse_mask = np.ones(ref_diffuseness.shape, dtype=bool)

        valid_mask = (energy_mask & direction_mask_ref & diffuse_mask)

        if not np.any(valid_mask):
            # If no valid time-frequency points, return 0
            return 0.0

        # Compute cosine similarity
        cosine_sim = np.zeros_like(dot_products)
        cosine_sim[valid_mask] = dot_products[valid_mask] / (ref_norm[valid_mask] * target_norm[valid_mask])

        # Clamp cosine similarity to avoid numerical issues
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Compute weights using reference signal only
        weights = np.ones_like(ref_energy)  # Default weights
        if self.energy_weighting:
            weights *= ref_energy
        if self.diffuseness_weighting:
            weights *= (1.0 - ref_diffuseness)
        
        # Apply validity mask to weights
        weights = weights * valid_mask.astype(float)
        
        # Compute weighted cosine loss (1 - cosine_similarity)
        cosine_loss = 1.0 - cosine_sim
        return np.mean(cosine_loss * weights)

    def estimate_direction_of_arrival(self, audio_signal: np.ndarray):
        """
        Estimate direction of arrival from a single audio signal.
        
        Parameters:
        -----------
        audio_signal : ndarray
            Input audio signal of shape (n_samples, 4) in Ambix format [W, Y, Z, X]
            
        Returns:
        --------
        azimuth : float
            Azimuth angle in degrees (-180 to 180, where 0° is front, 90° is left)
        elevation : float
            Elevation angle in degrees (-90 to 90, where 0° is horizontal, 90° is up)
        """
        # Extract STFT frames
        stft_frames = self._stft(audio_signal)

        rand_azi = np.degrees(np.random.uniform(0, 2 * np.pi))  # azimuth in [0, 360)
        rand_ele = np.degrees(np.arccos(np.random.uniform(-1, 1))) - 90  # elevation in [-90, 90]

        # Extract DirAC parameters
        intensity_vectors, diffuseness, energy = self._extract_dirac_parameters(stft_frames)

        # optional masking
        if self.energy_threshold is not None:
            energy_mask = energy > self.energy_threshold
        else:
            energy_mask = np.ones(energy.shape, dtype=bool)
        if self.diffuseness_threshold is not None:
            diffuseness_mask = diffuseness < self.diffuseness_threshold
        else:
            diffuseness_mask = np.ones(diffuseness.shape, dtype=bool)
        valid_mask = energy_mask & diffuseness_mask

        if not np.any(valid_mask):
            return rand_azi, rand_ele

        # optional weighting
        weights = np.ones_like(energy)  # Default weights
        if self.energy_weighting:
            weights *= energy
        if self.diffuseness_weighting:
            weights *= (1.0 - diffuseness)
        weights *= valid_mask.astype(float)  # Apply validity mask
        total_weight = np.sum(weights)
        if total_weight == 0:
            return rand_azi, rand_ele
        
        # Compute average intensity vector using weights
        avg_intensity = np.sum(intensity_vectors * weights[:, :, np.newaxis], axis=(0, 1)) / total_weight
        
        # Calculate weighted variance of intensity vectors
        # Variance = E[(X - μ)²] where μ is the weighted mean (avg_intensity)
        squared_diff = (intensity_vectors - avg_intensity[np.newaxis, np.newaxis, :])**2
        weighted_squared_diff = squared_diff * weights[:, :, np.newaxis]
        variance = np.sum(weighted_squared_diff, axis=(0, 1)) / total_weight
        
        # Total variance (sum of variances across all dimensions)
        avg_intensity_variance = np.sum(variance)

        # Extract XYZ components
        x, y, z = avg_intensity[0], avg_intensity[1], avg_intensity[2]
        
        # Convert to spherical coordinates
        # Azimuth: angle in XY plane from positive X axis (front)
        # In Ambix/audio convention: 0° = front, 90° = left, -90° = right
        azimuth_rad = np.arctan2(y, x)
        azimuth_deg = np.degrees(azimuth_rad)
        
        # Elevation: angle from XY plane towards positive Z axis (up)
        # Range: -90° to +90°
        r_xy = np.sqrt(x**2 + y**2)
        elevation_rad = np.arctan2(z, r_xy)
        elevation_deg = np.degrees(elevation_rad)


        return azimuth_deg, elevation_deg, avg_intensity_variance

def compute_spatial_consistency_loss(reference_audio: np.ndarray, 
                                   target_audio: np.ndarray,
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


def estimate_direction_of_arrival(audio_signal: np.ndarray,
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
    # Simple test
    print("Testing Spatial Consistency Calculator")
    
    # Create dummy test data
    np.random.seed(42)
    n_samples = 8192
    fs = 24000  # Sample rate
    
    # Create reference audio (random 4-channel Ambix signal: W, Y, Z, X)
    #reference = np.random.randn(n_samples, 4) * 0.1
    import librosa
    try:
        reference, reffs = librosa.load("/home/hagamper/dirac/p225_001_mic1_azi=134.83_zen=25.65.wav", sr=None, mono=False)
        if reffs != fs:
            import librosa
            reference = librosa.resample(reference, orig_sr=reffs, target_sr=fs)
            print(f"Resampled reference audio from {reffs} Hz to {fs} Hz")
            reference = reference[:,:n_samples].T  # Transpose to shape (n_samples, 4)
    except FileNotFoundError:
        print("Reference audio file not found. Using random data instead.")
        reference = np.random.randn(n_samples, 4) * 0.1

    print("\n--- Testing Direction of Arrival Estimation ---")
    
    # Test DOA estimation with the reference signal
    azimuth, elevation, avg_intensity_variance = estimate_direction_of_arrival(reference)
    print(f"Reference signal DOA: Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")
    
    # Create a test signal with known direction (45° left, 30° up)
    # This is a simplified simulation - in practice, proper ambisonic encoding would be used
    test_signal = np.random.randn(n_samples, 4) * 0.1
    # Simulate signal coming from 45° left (positive Y) and 30° up (positive Z)
    # Enhance Y and Z components relative to X component
    test_signal[:, 1] *= 2.0  # Y component (left)
    test_signal[:, 2] *= 1.0  # Z component (up)
    test_signal[:, 3] *= 0.5  # X component (front)
    
    azimuth_test, elevation_test, avg_intensity_variance = estimate_direction_of_arrival(test_signal)
    print(f"Test signal DOA: Azimuth = {azimuth_test:.2f}°, Elevation = {elevation_test:.2f}°")

    # test different sample cases
    test_cases = [
        "small perturbation",
        "rotation about y-axis",
        "rotation about z-axis",
        "identical signals"
    ]

    print("\n--- Testing Spatial Consistency Loss ---")

    for case in test_cases:
        if case == "small perturbation":
            target = reference + np.random.randn(n_samples, 4) * 0.01
        elif case == "rotation about y-axis":
            # Rotation about y-axis affects X and Z components (preserves lateral angle)
            # Rotation matrix for y-axis
            target = np.copy(reference)
            angle = 10 * np.pi / 180  # 10 degrees
            R_y = np.array([
                [np.cos(angle), 0, -np.sin(angle)],
                [0, 1, 0],
                [np.sin(angle), 0, np.cos(angle)]
            ])
            # Extract XYZ components (indices 3,1,2 in Ambix [W,Y,Z,X] -> [X,Y,Z])
            xyz = reference[:, [3, 1, 2]]  # Shape: (n_samples, 3) [X, Y, Z]
            # Apply rotation matrix
            xyz_rotated = np.dot(xyz, R_y.T)  
            # Put back into Ambix format [W,Y,Z,X]
            target[:, 1] = xyz_rotated[:, 1]  # Y component
            target[:, 2] = xyz_rotated[:, 2]  # Z component
            target[:, 3] = xyz_rotated[:, 0]  # X component
        elif case == "rotation about z-axis":
            # Rotation about z-axis affects X and Y components (changes lateral angle)
            # Rotation matrix for z-axis: R_z(θ) = [cos θ, -sin θ, 0; sin θ, cos θ, 0; 0, 0, 1]
            target = np.copy(reference)
            angle = 10 * np.pi / 180  # 10 degrees
            R_z = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            # Extract XYZ components (indices 3,1,2 in Ambix [W,Y,Z,X] -> [X,Y,Z])
            xyz = reference[:, [3, 1, 2]]  # Shape: (n_samples, 3) [X, Y, Z]
            # Apply rotation matrix
            xyz_rotated = np.dot(xyz, R_z.T)  
            # Put back into Ambix format [W,Y,Z,X]
            target[:, 1] = xyz_rotated[:, 1]  # Y component
            target[:, 2] = xyz_rotated[:, 2]  # Z component
            target[:, 3] = xyz_rotated[:, 0]  # X component
        elif case == "identical signals":
            target = np.copy(reference)
        
        loss = compute_spatial_consistency_loss(reference, target)
        print(f"Case '{case}': Spatial Consistency Loss = {loss:.4f}")
        # test lateral only
        loss_lateral = compute_spatial_consistency_loss(reference, target, lateral_only=True)
        print(f"Case '{case}' (lateral only): Spatial Consistency Loss = {loss_lateral:.4f}")

        # Test DOA estimation with the reference signal
        azimuth, elevation, avg_intensity_variance = estimate_direction_of_arrival(target)
        print(f"Target signal DOA: Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")
    
    # Test without validity mask
    #azimuth_no_mask, elevation_no_mask = estimate_direction_of_arrival(test_signal)
    #print(f"Test signal DOA (no mask): Azimuth = {azimuth_no_mask:.2f}°, Elevation = {elevation_no_mask:.2f}°")

    print("Test completed.")
