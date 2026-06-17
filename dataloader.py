from torch.utils.data import DataLoader
import webdataset as wds
import glob
import torch
import torchaudio
import io

num_samples = 240000

def preprocess(sample):   
    try:
        audio_bytes = sample["wav"]
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