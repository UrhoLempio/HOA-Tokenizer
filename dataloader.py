from torch.utils.data import DataLoader
import webdataset as wds
import glob
import torch
import io

# Global setting for which audio loader to use
AUDIO_LOADER = 'torchaudio'  # 'torchaudio' or 'soundfile'

def set_audio_loader(loader_type: str):
    """Set which audio loader to use: 'torchaudio' or 'soundfile'"""
    global AUDIO_LOADER
    if loader_type not in ('torchaudio', 'soundfile'):
        raise ValueError(f"audio_loader must be 'torchaudio' or 'soundfile', got {loader_type}")
    AUDIO_LOADER = loader_type

num_samples = 240000

def preprocess(sample):   
    try:
        audio_bytes = sample["wav"]
        
        if AUDIO_LOADER == 'soundfile':
            import soundfile as sf
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))  # numpy array [T, C] or [T]
            # Convert to torch tensor and extract first channel
            if audio_np.ndim == 1:
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, T]
            else:
                audio = torch.from_numpy(audio_np.T).float()  # [C, T]
                if audio.shape[0] > 1:
                    audio = audio[0:1, :]  # Keep first channel as [1, T]
        else:
            import torchaudio
            audio, sr = torchaudio.load(io.BytesIO(audio_bytes))  # [C, T]
            # ✅ Convert to mono (if needed)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

        # ✅ Crop or pad
        length = audio.shape[1]

        if length > num_samples:
            start = torch.randint(0, length - num_samples, (1,)).item()
            audio = audio[:, start:start + num_samples]

        elif length < num_samples:
            pad = num_samples - length
            audio = torch.nn.functional.pad(audio, (0, pad))

        # ✅ if length == num_samples → do nothing
        else:
            pad = num_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad))

        return {"audio": audio}
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def get_dataloaders(
    train_dir,
    val_dir,
    train_batch_size=2,
    train_num_workers=0,
    val_batch_size=2,
    val_num_workers=0,
    pin_memory=True,
):
    train_shard_paths = glob.glob(f"{train_dir}/*.tar")
    if not train_shard_paths:
        raise FileNotFoundError(
            f"No .tar files found in {train_dir}. "
            f"Check that the path exists and contains .tar shards."
        )
    
    train_dataset = (
        wds.WebDataset(train_shard_paths, shardshuffle=1000)
        .map(preprocess)
        .shuffle(2000)
        .repeat()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    test_shard_paths = glob.glob(f"{val_dir}/*.tar")
    if not test_shard_paths:
        raise FileNotFoundError(
            f"No .tar files found in {val_dir}. "
            f"Check that the path exists and contains .tar shards."
        )

    val_dataset = (
        wds.WebDataset(test_shard_paths, shardshuffle=False)
        .map(preprocess)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader