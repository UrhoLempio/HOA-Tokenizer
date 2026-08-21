import argparse
import json
import os
from pathlib import Path
import psutil

import torch
import torchaudio
import torch.distributed as dist
from model import HOA_WavTokenizer
from discriminator import DACDiscriminator, MultiPeriodDiscriminator, MultiResolutionDiscriminator
from loss import MelSpecReconstructionLoss, GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss, DACGANLoss
import auraloss
from angular_error import angular_error
from dataloader import get_dataloaders, set_audio_loader, set_target_channels
from SpatialConsistency import SpatialConsistency, estimate_direction_of_arrival
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# TQDM switch since cluster terminals may not support it. Use config to set this.
USE_TQDM = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

def setup_ddp():
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def validate(model: torch.nn.Module, 
             val_loader: torch.utils.data.DataLoader, 
             mel_loss_fn: torch.nn.Module, 
             mrstft_loss_fn: torch.nn.Module,
             bandwidth: float,
             device: torch.device) -> tuple:
    """
    Convenience function to validate the model on the validation set.
    This function runs the model on the validation set and computes the average mel loss.
    It also returns a reconstructed audio sample and its source filename.

    Parameters:
    -----------
    model: torch.nn.Module
            the generator model to validate
    val_loader: torch.utils.data.DataLoader
            the validation dataloader
    mel_loss_fn: torch.nn.Module
            the mel loss function to use for validation
    mrstft_loss_fn: torch.nn.Module
            the multi-resolution STFT loss function to use for validation
    device: torch.device
            the device to run the validation on
    Returns:
    -----------
    val_loss: float
            the average validation loss over the validation set
    mrstft_loss: float
            the average multi-resolution STFT loss over the validation set
    angular_error: float
            the average angular error over the validation set
    sample_audio: torch.Tensor
            a sample audio from the validation set
    sample_source: str
            the source of the sample audio
    """
    model.eval()

    
    val_losses = []
    mrstft_losses = []
    angular_errors = []
    sample_audio = None
    sample_source = "unknown"
    with torch.no_grad():
        # Change to "batch in val_loader" when full validation is needed.
        # For now we just want to check if the validation loop runs and produces reasonable output.
        # This is a speed hack to avoid running the full validation which can be time consuming.
        # TODO stft distance, angular error and plot to tensorboard
        
        for i, batch in enumerate(val_loader):
            if i > 20:
                break   
            audio_input = batch["audio"].to(device)
            out = model(audio_input, bandwidth=bandwidth)
            audio_hat = out["audio"]

            az_hat, el_hat, _ = estimate_direction_of_arrival(audio_hat)
            az_input, el_input, _ = estimate_direction_of_arrival(audio_input)            

            val_losses.append(mel_loss_fn(audio_hat, audio_input).item())
            mrstft_losses.append(mrstft_loss_fn(audio_hat, audio_input).item())
            angular_errors.append(angular_error(az_hat, el_hat, az_input, el_input).mean().item())

            if sample_audio is None:
                sample_audio = audio_hat[0].detach().cpu()
                sample_source = batch["source"][0]

    model.train()

    if not val_losses:
        raise RuntimeError("Validation loader is empty.")

    return sum(val_losses) / len(val_losses), sum(mrstft_losses) / len(mrstft_losses), sum(angular_errors) / len(angular_errors), sample_audio, sample_source


def load_config(config_path: Path):
    """
    Load the configuration from a json or yaml file.

    Parameters:
    -----------
    config_path: Path
            The path to the configuration file

    Returns:
    --------
    dict
            The loaded configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        if config_path.suffix in {".yaml", ".yml"}:
            if not _HAS_YAML:
                raise RuntimeError(
                    "YAML config support requires PyYAML. Install it with 'pip install pyyaml'."
                )
            return yaml.safe_load(f)
        return json.load(f)


def config_int(config, section, key, default):
    """Helper function to guarantee that the config value is an integer."""
    value = config.get(section, {}).get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected integer for {section}.{key}, got {value!r}")


def config_float(config, section, key, default):
    """Helper function to guarantee that the config value is a float."""
    value = config.get(section, {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float for {section}.{key}, got {value!r}")


def config_bool(config, section, key, default):
    """Helper function to guarantee that the config value is a boolean."""
    value = config.get(section, {}).get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"Expected boolean for {section}.{key}, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the HOA WavTokenizer from a config file.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the JSON or YAML config file",
    )
    return parser.parse_args()


def main(config):

    global USE_TQDM
    use_tqdm = config.get("env", {}).get("use_tqdm", True)
    USE_TQDM = use_tqdm
    if USE_TQDM:
        from tqdm import tqdm

    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    checkpoint_dir = Path(config["io"].get("checkpoint_dir", "./checkpoints"))
    samples_dir = Path(config["io"].get("generator_samples_dir", "./generator_samples"))
    val_samples_dir = Path(config["io"].get("val_samples_dir", "./val_samples"))
    logs_dir = Path(config["io"].get("log_dir", "./logs"))
    bandwidth = config_float(config, "model", "bandwidth", 6.6)
    in_channels = config_int(config, "model", "in_channels", 4)
    train_batch_size = config_int(config, "training", "train_batch_size", 2)
    val_batch_size = config_int(config, "training", "val_batch_size", 2)
    train_num_workers = config_int(config, "training", "train_num_workers", 0)
    val_num_workers = config_int(config, "training", "val_num_workers", 0)
    pin_memory = config_bool(config, "training", "pin_memory", True)
    max_steps = config_int(config, "training", "max_steps", 50000)
    pretrain_mel_steps = config_int(config, "training", "pretrain_mel_steps", 0)
    mel_loss_coeff = config_float(config, "training", "mel_loss_coeff", 45.0)
    mrd_loss_coeff = config_float(config, "training", "mrd_loss_coeff", 1.0)
    spatial_loss_coeff = config_float(config, "training", "spatial_loss_coeff", 0.01)
    # One-time spatial loss boost: set to this value when step >= spatial_boost_step
    spatial_boost_applied = False
    spatial_boost_step = 100000
    spatial_boost_value = 5.0
    commit_loss_coeff = config_float(config, "training", "commit_loss_coeff", 1000.0)
    grad_clip_norm = config_float(config, "training", "grad_clip_norm", 1.0)
    spatial_loss_every = config_int(config, "training", "spatial_loss_every", 5)
    val_every = config_int(config, "training", "val_every", 2000)
    save_every = config_int(config, "training", "save_every", 5000)
    sample_every = config_int(config, "training", "sample_every", save_every)
    max_checkpoints = config_int(config, "training", "max_checkpoints", 5)
    learning_rate = config_float(config, "training", "lr", 2e-4)

    # Device setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    print(
        f"rank={rank}, "
        f"local_rank={local_rank}, "
        f"world_size={world_size}",
        flush=True
    )

    # Set audio loader to torchaudio or soundfile based on config
    audio_loader = config.get("env", {}).get("audio_loader", "torchaudio")
    set_audio_loader(audio_loader)

    # Set target channels for audio loading
    set_target_channels(in_channels)

    # Create necessary directories
    for path in (checkpoint_dir, samples_dir, val_samples_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    if rank == 0:
        writer = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dir,
        val_dir,
        train_batch_size=train_batch_size,
        train_num_workers=train_num_workers,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        pin_memory=pin_memory,
        target_channels=in_channels,
    )

    # Model
    model = HOA_WavTokenizer(in_channels=in_channels).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        )

    # Optional print of model parameters (74.69M parameters)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    if rank == 0:
        total_params = count_parameters(model)
        print(f"Model parameters: {total_params / 1e6:.2f}M") 

    # Discriminators
    disc_mpd = MultiPeriodDiscriminator(in_channels=in_channels).to(device)
    disc_mpd = DDP(disc_mpd, device_ids=[local_rank], output_device=local_rank)
    disc_mrd = MultiResolutionDiscriminator(in_channels=in_channels).to(device)
    disc_mrd = DDP(disc_mrd, device_ids=[local_rank], output_device=local_rank)
    disc_dac = DACDiscriminator(in_channels=in_channels).to(device)
    disc_dac = DDP(disc_dac, device_ids=[local_rank], output_device=local_rank)
    discriminators = [disc_mpd, disc_mrd, disc_dac]

    # Losses 
    mel_loss_fn = MelSpecReconstructionLoss(sample_rate=24000).to(device)
    gen_loss_fn = GeneratorLoss().to(device)
    spatial_consistency_fn = SpatialConsistency(sample_rate=24000)
    disc_loss_fn = DiscriminatorLoss().to(device)
    feat_match_loss_fn = FeatureMatchingLoss().to(device)
    dac_loss = DACGANLoss(disc_dac).to(device)

    # Metrics
    mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss()

    # Optimizers
    opt_gen = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    disc_params = []
    for d in discriminators:    
        disc_params += list(d.parameters())
    opt_disc = torch.optim.AdamW(disc_params, lr=learning_rate)

    # AMP
    use_amp = True
    scaler = GradScaler(device.type) if use_amp else None

    # Checkpoint loading
    resume_path = None
    for candidate in sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if candidate.name.startswith("checkpoint_best"):
            continue
        resume_path = candidate
        break
    best_val_loss = float("inf")

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)

        model.module.load_state_dict(ckpt["model"])
        disc_mpd.module.load_state_dict(ckpt["disc_mpd"])
        disc_mrd.module.load_state_dict(ckpt["disc_mrd"])
        disc_dac.module.load_state_dict(ckpt["disc_dac"])

        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_disc.load_state_dict(ckpt["opt_disc"])

        global_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if rank == 0:
            print(f"✅ Resumed from step {global_step}")

    else:
        global_step = 0
    
    #############################
    if rank == 0:
        print("Starting training...")
    #############################

    if USE_TQDM:
        pbar = tqdm(total=max_steps)
    else:
        pbar = None

    batch = next(iter(train_loader))

    # Sanity check for batch shape
    if rank == 0:
        print(f"batch['audio'].shape: {batch['audio'].shape} Expecting [B, C, T] with C={in_channels} channels")

    while global_step < max_steps:  
        for batch in train_loader:
            #print(f"Entered training loop step {global_step}", flush=True)
            audio_input = batch["audio"].to(device)  # [B, C, T]

            # match Lightning behavior
            train_discriminator = global_step >= pretrain_mel_steps
            loss_disc = torch.tensor(0.0, device=device)

            # Apply one-time spatial loss coefficient boost at configured milestone
            if (not spatial_boost_applied) and global_step >= spatial_boost_step:
                spatial_loss_coeff = float(spatial_boost_value)
                spatial_boost_applied = True
                if rank == 0:
                    print(f"🔼 Boosted spatial_loss_coeff to {spatial_boost_value} at step {global_step}", flush=True)

            # ==================================================
            # DISCRIMINATOR STEP
            # ==================================================
            if train_discriminator:
                opt_disc.zero_grad(set_to_none=True)

                with torch.no_grad():
                    out = model(audio_input, bandwidth=bandwidth)
                    codes = out["codes"]
                    audio_hat = out["audio"]
                loss_dac_total = 0.0
                loss_mp_total = 0.0
                loss_mrd_total = 0.0
                with autocast(device_type=device.type, enabled=use_amp):
                    # TODO change the discriminators for four channels instead
                    loss_dac_total += dac_loss.discriminator_loss(audio_hat, audio_input)
                    
                    real_mp, gen_mp, _, _ = disc_mpd(y=audio_input, y_hat=audio_hat)
                    loss_mp, loss_mp_real, _ = disc_loss_fn(
                        disc_real_outputs=real_mp,
                        disc_generated_outputs=gen_mp,
                    )
                    loss_mp = loss_mp / len(loss_mp_real)
                    loss_mp_total += loss_mp

                    real_mrd, gen_mrd, _, _ = disc_mrd(y=audio_input, y_hat=audio_hat)
                    loss_mrd, loss_mrd_real, _ = disc_loss_fn(
                        disc_real_outputs=real_mrd,
                        disc_generated_outputs=gen_mrd,
                    )
                    loss_mrd = loss_mrd / len(loss_mrd_real)
                    loss_mrd_total += loss_mrd

                    loss_dac = loss_dac_total
                    loss_mp = loss_mp_total
                    loss_mrd = loss_mrd_total

                    loss_disc = loss_mp + mrd_loss_coeff * loss_mrd + loss_dac

                if scaler is not None:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(opt_disc)
                    torch.nn.utils.clip_grad_norm_(disc_params, grad_clip_norm)
                    scaler.step(opt_disc)
                    scaler.update()
                else:
                    loss_disc.backward()
                    torch.nn.utils.clip_grad_norm_(disc_params, grad_clip_norm)
                    opt_disc.step()

            # ==================================================
            # GENERATOR STEP
            # ==================================================
            opt_gen.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(audio_input, bandwidth=bandwidth)
                audio_hat = out["audio"]
                commit_loss = out["commit_loss"]
                if global_step % 10 == 0:
                    if not torch.isfinite(commit_loss):
                        raise RuntimeError(f"BAD COMMIT LOSS at step {global_step}")
                        
                    if not torch.isfinite(audio_hat).all():
                        raise RuntimeError(f"BAD AUDIO_HAT at step {global_step}")

                if train_discriminator:
                    loss_dac_1_total = 0.0
                    loss_dac_2_total = 0.0
                    loss_gen_mp_total = 0.0
                    loss_fm_mp_total = 0.0
                    loss_gen_mrd_total = 0.0
                    loss_fm_mrd_total = 0.0

                    loss_dac_1, loss_dac_2 = dac_loss.generator_loss(audio_hat, audio_input)
                    loss_dac_1_total += loss_dac_1
                    loss_dac_2_total += loss_dac_2

                    _, gen_mp, fmap_rs_mp, fmap_gs_mp = disc_mpd(y=audio_input, y_hat=audio_hat)
                    loss_gen_mp, list_loss_gen_mp = gen_loss_fn(gen_mp)
                    loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                    loss_gen_mp_total += loss_gen_mp

                    loss_fm_mp = feat_match_loss_fn(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                    loss_fm_mp_total += loss_fm_mp

                    _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = disc_mrd(y=audio_input, y_hat=audio_hat)
                    loss_gen_mrd, list_loss_gen_mrd = gen_loss_fn(gen_mrd)
                    loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                    loss_gen_mrd_total += loss_gen_mrd

                    loss_fm_mrd = feat_match_loss_fn(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
                    loss_fm_mrd_total += loss_fm_mrd
                    loss_dac_1 = loss_dac_1_total
                    loss_dac_2 = loss_dac_2_total
                    loss_gen_mp = loss_gen_mp_total
                    loss_fm_mp = loss_fm_mp_total
                    loss_gen_mrd = loss_gen_mrd_total
                    loss_fm_mrd = loss_fm_mrd_total

                else:
                    # pretraining phase
                    loss_gen_mp = 0
                    loss_gen_mrd = 0
                    loss_fm_mp = 0
                    loss_fm_mrd = 0
                    loss_dac_1 = 0
                    loss_dac_2 = 0

            if not torch.isfinite(audio_hat).all():
                raise RuntimeError(
                    f"audio_hat became non-finite at step {global_step}"
                )
            # Mel loss
            with autocast(device_type=device.type, enabled=False):
                mel_loss = mel_loss_fn(audio_hat.float(), audio_input.float())

            # Total generator loss
            spatial_loss = torch.zeros((), device=device, dtype=torch.float32)
            mask_ratio = torch.zeros((), device=device, dtype=torch.float32)

            if spatial_loss_coeff != 0.0 and (spatial_loss_every <= 1 or global_step % spatial_loss_every == 0):
                ref_audio = audio_input.transpose(1, 2)
                gen_audio = audio_hat.transpose(1, 2)

                spatial_loss, mask_ratio = spatial_consistency_fn.compute_spatial_consistency(
                    ref_audio,
                    gen_audio,
                )

            loss_gen = (
                loss_gen_mp
                + mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + mrd_loss_coeff * loss_fm_mrd
                + mel_loss_coeff * mel_loss
                + commit_loss_coeff * commit_loss
                + loss_dac_1
                + loss_dac_2
                + spatial_loss_coeff * spatial_loss
            )

            # DEBUG CHECKS
            loss_components = {
                "loss_gen_mp": (loss_gen_mp, 1.0),
                "loss_gen_mrd": (loss_gen_mrd, mrd_loss_coeff),
                "loss_fm_mp": (loss_fm_mp, 1.0),
                "loss_fm_mrd": (loss_fm_mrd, mrd_loss_coeff),
                "mel_loss": (mel_loss, mel_loss_coeff),
                "commit_loss": (commit_loss, commit_loss_coeff),
                "loss_dac_1": (loss_dac_1, 1.0),
                "loss_dac_2": (loss_dac_2, 1.0),
                "spatial_loss": (spatial_loss, spatial_loss_coeff),
            }

            if not torch.isfinite(loss_gen):
                debug_values = {}
                for name, (value, coefficient) in loss_components.items():
                    value_tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
                    weighted_value = value_tensor * coefficient
                    debug_values[name] = {
                        "raw": value_tensor.detach().float().item(),
                        "coefficient": coefficient,
                        "weighted": weighted_value.detach().float().item(),
                        "finite": bool(torch.isfinite(value_tensor).all()),
                        "weighted_finite": bool(torch.isfinite(weighted_value).all()),
                    }

                print(
                    f"NON-FINITE LOSS DEBUG at step {global_step}, rank {rank}: "
                    f"loss_gen={loss_gen.detach().float().item()} | "
                    f"components={debug_values}",
                    flush=True,
                )

            if not torch.isfinite(mel_loss):
                raise RuntimeError(
                    f"mel_loss became non-finite at step {global_step}"
                )

            if not torch.isfinite(spatial_loss):
                raise RuntimeError(
                    f"spatial_loss became non-finite at step {global_step}"
                )

            if not torch.isfinite(commit_loss):
                raise RuntimeError(
                    f"commit_loss became non-finite at step {global_step}"
                )

            if not torch.isfinite(loss_gen):
                raise RuntimeError(
                    f"loss_gen became non-finite at step {global_step}"
                )

            if scaler is not None:
                scaler.scale(loss_gen).backward()
                scaler.unscale_(opt_gen)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt_gen)
                scaler.update()
            else:
                loss_gen.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt_gen.step()

            # ==================================================
            # LOGGING & CHECKPOINTS
            # ==================================================
            if rank == 0 and global_step % 10 == 0:
                print(
                    f"[{global_step}] "
                    f"Gen: {loss_gen.item():.4f} | "
                    f"Disc: {loss_disc.item():.4f} | "
                    f"Mel: {mel_loss.item():.4f} | "
                    f"Commit: {commit_loss.item():.3e} | "
                    f"Spatial: {spatial_loss.item():.4f} | ",
                    flush=True
                )
                writer.add_scalar("loss/train_gen", loss_gen.item(), global_step)
                writer.add_scalar("loss/train_disc", loss_disc.item(), global_step)
                writer.add_scalar("loss/mel", mel_loss.item(), global_step)
                writer.add_scalar("loss/commit", commit_loss.item(), global_step)
                writer.add_scalar("loss/gen_mp", loss_gen_mp, global_step)
                writer.add_scalar("loss/gen_mrd", loss_gen_mrd, global_step)
                writer.add_scalar("loss/discriminators/mpd", loss_mp.item(), global_step)
                writer.add_scalar("loss/discriminators/mrd", loss_mrd.item(), global_step)
                writer.add_scalar("loss/discriminators/dac", loss_dac.item(), global_step)
                writer.add_scalar("loss/feature_matching/mpd", loss_fm_mp.item(), global_step)
                writer.add_scalar("loss/feature_matching/mrd", loss_fm_mrd.item(), global_step)
                            
            if rank == 0 and spatial_loss_coeff != 0.0 and (spatial_loss_every <= 1 or global_step % spatial_loss_every == 0):
                writer.add_scalar("loss/spatial", spatial_loss.item(), global_step)
                writer.add_scalar("debug/mask_ratio", mask_ratio.item(), global_step)
                writer.add_scalar("loss/weighted/mel", (mel_loss_coeff * mel_loss).item(), global_step)
                writer.add_scalar("loss/weighted/commit", (commit_loss_coeff * commit_loss).item(), global_step)
                writer.add_scalar("loss/weighted/spatial", (spatial_loss_coeff * spatial_loss).item(), global_step)



            if rank == 0 and global_step % 200 == 0:    
                writer.flush()
            if rank == 0 and global_step % 100 == 0:
                for q in range(codes.shape[0]):
                    active_codes = torch.unique(codes[q]).numel()
                
                    writer.add_scalar(
                    f"vq/active_codes_q{q}",
                    active_codes,
                    global_step,
                    )
                    
                    writer.add_scalar(
                    f"vq/utilization_q{q}",
                    active_codes / 1024.0,
                    global_step,
                    )
                    
                    writer.add_scalar(
                    f"vq/dead_codes_q{q}",
                    1024 - active_codes,
                    global_step,
                    )

            if global_step != 0 and global_step % val_every == 0:                
                val_loss, mrstft_loss, angular_error, val_sample, val_reference_fname = validate(model, val_loader, mel_loss_fn, mrstft_loss_fn, bandwidth, device)
                if rank == 0:
                    writer.add_scalar("loss/validation/mel", val_loss, global_step)
                    writer.add_scalar("loss/validation/mrstft", mrstft_loss, global_step)
                    writer.add_scalar("loss/validation/angular", angular_error, global_step)
                    writer.flush()
                    print(f"[{global_step}] Val mel: {val_loss:.4f} MRSTFT: {mrstft_loss:.4f} Angular: {angular_error:.4f}", flush=True)
                    torchaudio.save(
                        str(val_samples_dir / f"val_{global_step}_{val_reference_fname}.wav"),
                        val_sample,
                        24000,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = {
                            "model": model.module.state_dict(),
                            "disc_mpd": disc_mpd.module.state_dict(),
                            "disc_mrd": disc_mrd.module.state_dict(),
                            "disc_dac": disc_dac.module.state_dict(),
                            "opt_gen": opt_gen.state_dict(),
                            "opt_disc": opt_disc.state_dict(),
                            "step": global_step,
                            "best_val_loss": best_val_loss,
                        }
                        torch.save(best_checkpoint, str(checkpoint_dir / f"checkpoint_best.pt"))
                        with open(
                            checkpoint_dir / "checkpoint_best_info.txt",
                            "w"
                        ) as f:
                            f.write(
                                f"step={global_step}\n"
                                f"val_loss={best_val_loss}\n"
                            )
                                
                        print(
                            f"✅ New best validation checkpoint "
                            f"(step {global_step}, "
                            f"val={best_val_loss:.4f})",
                            flush=True,
                            )
                dist.barrier()  # Ensure all processes have completed before continuing training

            if rank == 0 and global_step != 0 and global_step % save_every == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "disc_mpd": disc_mpd.module.state_dict(),
                    "disc_mrd": disc_mrd.module.state_dict(),
                    "disc_dac": disc_dac.module.state_dict(),
                    "opt_gen": opt_gen.state_dict(),
                    "opt_disc": opt_disc.state_dict(),
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, str(checkpoint_dir / f"checkpoint_{global_step}.pt"))

                all_ckpts = sorted(
                    [p for p in checkpoint_dir.glob("checkpoint_*.pt") if not p.name.startswith("checkpoint_best")],
                    key=os.path.getmtime,
                )

                if len(all_ckpts) > max_checkpoints:
                    for ck in all_ckpts[:-max_checkpoints]:
                        ck.unlink()

                print(f"✅ Saved checkpoint at step {global_step}", flush=True)

            if rank == 0 and global_step != 0 and global_step % sample_every == 0:
                torchaudio.save(
                    str(samples_dir / f"sample_{global_step}.wav"),
                    audio_hat[0].detach().cpu(),
                    24000,
                )

            if rank == 0 and global_step != 0 and global_step % 100 == 0:
                total = 0
                parent = psutil.Process()

                for p in [parent] + parent.children(recursive=True):
                    try:
                        total += p.memory_info().rss
                    except:
                        pass

                print(
                    f"TOTAL RAM: {total/1024**3:.2f} GB"
                )

            global_step += 1

            # ==================================================
            # PROGRESS BAR UPDATE
            # ==================================================
            if USE_TQDM:
                pbar.update(1)
                if rank == 0 and global_step % 100 == 0:
                    pbar.set_description(
                        f"G:{loss_gen.item():.2f} D:{loss_disc.item():.2f}"
                    )
    if rank == 0:
        writer.close()
        print("Training completed successfully!")
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(Path(args.config_path))
    main(config)