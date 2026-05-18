from torch.utils.data import DataLoader
from dataset import AudioDataset
#from torch.utils.data.distributed import DistributedSampler



def get_dataloaders(train_dir, val_dir):
    train_dataset = AudioDataset(
        train_dir,
        num_samples=240000,
        sampling_rate=24000
    )

    val_dataset = AudioDataset(
        val_dir,
        num_samples=240000,
        sampling_rate=24000
    )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=40,
    #     shuffle=True,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    #     persistent_workers=True
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    return train_loader, val_loader