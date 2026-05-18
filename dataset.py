import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root_dir, num_samples=240000, sampling_rate=24000):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.sr = sampling_rate

        # collect all audio files
        exts = [".wav", ".flac", ".mp3"]
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if any(f.endswith(e) for e in exts)
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def load_audio(self, path):
        audio, sr = torchaudio.load(path)  # (C, T)

        # convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)

        return audio  # (1, T)

    def random_crop(self, audio):
        T = audio.shape[1]

        if T >= self.num_samples:
            start = random.randint(0, T - self.num_samples)
            return audio[:, start:start + self.num_samples]

        # pad if too short
        pad_size = self.num_samples - T
        audio = torch.nn.functional.pad(audio, (0, pad_size))
        return audio

    def __getitem__(self, idx):
        path = self.files[idx]

        audio = self.load_audio(path)
        audio = self.random_crop(audio)

        return {
            "audio": audio,   # shape: (1, 240000)
            "path": path
        }