import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from model import HOA_WavTokenizer
from discriminator import DACDiscriminator, MultiPeriodDiscriminator, MultiResolutionDiscriminator
from loss import MelSpecReconstructionLoss, GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss, DACGANLoss
from dataloader import get_dataloaders
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def validate(model, discriminators, val_loader, mel_loss_fn, device):
    model.eval()
    for d in discriminators:
        d.eval()

    val_losses = []
    sample_audio = None

    with torch.no_grad():
        # Change to "batch in val_loader" when full validation is needed. 
        # For now we just want to check if the validation loop runs and produces reasonable output.
        # This is a speed hack to avoid running the full validation which can be time consuming.
        for i, batch in enumerate(val_loader):
            if i > 20:
                break   
            audio_input = batch["audio"].to(device)
            out = model(audio_input)
            audio_hat = out["audio"]
            val_losses.append(mel_loss_fn(audio_hat, audio_input).item())
            if sample_audio is None:
                sample_audio = audio_hat[0].detach().cpu()

    model.train()
    for d in discriminators:
        d.train()

    if not val_losses:
        raise RuntimeError("Validation loader is empty.")

    return sum(val_losses) / len(val_losses), sample_audio


def load_config(config_path: Path):
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
    value = config.get(section, {}).get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected integer for {section}.{key}, got {value!r}")


def config_float(config, section, key, default):
    value = config.get(section, {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float for {section}.{key}, got {value!r}")


def config_bool(config, section, key, default):
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
    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]

    train_batch_size = config_int(config, "training", "train_batch_size", 2)
    val_batch_size = config_int(config, "training", "val_batch_size", 2)
    train_num_workers = config_int(config, "training", "train_num_workers", 0)
    val_num_workers = config_int(config, "training", "val_num_workers", 0)
    pin_memory = config_bool(config, "training", "pin_memory", True)

    checkpoint_dir = Path(config["io"].get("checkpoint_dir", "./checkpoints"))
    samples_dir = Path(config["io"].get("generator_samples_dir", "./generator_samples"))
    val_samples_dir = Path(config["io"].get("val_samples_dir", "./val_samples"))
    logs_dir = Path(config["io"].get("log_dir", "./logs"))
    for path in (checkpoint_dir, samples_dir, val_samples_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))

    if config["training"].get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["training"]["device"]

    train_loader, val_loader = get_dataloaders(
        train_dir,
        val_dir,
        train_batch_size=train_batch_size,
        train_num_workers=train_num_workers,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = HOA_WavTokenizer().to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params / 1e6:.2f}M") # 71.69M Model Parameters

    # Discriminators
    disc_mpd = MultiPeriodDiscriminator().to(device)
    disc_mrd = MultiResolutionDiscriminator().to(device)
    disc_dac = DACDiscriminator().to(device)
    discriminators = [disc_mpd, disc_mrd, disc_dac]

    # Losses 
    mel_loss_fn = MelSpecReconstructionLoss(sample_rate=24000).to(device)
    gen_loss_fn = GeneratorLoss().to(device)
    disc_loss_fn = DiscriminatorLoss().to(device)
    feat_match_loss_fn = FeatureMatchingLoss().to(device)
    dac_loss = DACGANLoss(disc_dac).to(device)

    # Optimizers
    learning_rate = config_float(config, "training", "lr", 2e-4)
    opt_gen = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    disc_params = []
    for d in discriminators:    
        disc_params += list(d.parameters())
    opt_disc = torch.optim.AdamW(disc_params, lr=learning_rate)

    # Checkpoint loading
    resume_path = checkpoint_dir / "checkpoint_latest.pt"
    best_val_loss = float("inf")

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        disc_mpd.load_state_dict(ckpt["disc_mpd"])
        disc_mrd.load_state_dict(ckpt["disc_mrd"])
        disc_dac.load_state_dict(ckpt["disc_dac"])

        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_disc.load_state_dict(ckpt["opt_disc"])

        global_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        print(f"✅ Resumed from step {global_step}")

    else:
        global_step = 0



    max_steps = config_int(config, "training", "max_steps", 50000)
    pretrain_mel_steps = config_int(config, "training", "pretrain_mel_steps", 0)
    mel_loss_coeff = config_float(config, "training", "mel_loss_coeff", 45.0)
    mrd_loss_coeff = config_float(config, "training", "mrd_loss_coeff", 1.0)
    val_every = config_int(config, "training", "val_every", 2000)
    save_every = config_int(config, "training", "save_every", 5000)
    sample_every = config_int(config, "training", "sample_every", save_every)
    max_checkpoints = config_int(config, "training", "max_checkpoints", 5)

    print("Starting training...")




    pbar = tqdm(total=max_steps)

    batch = next(iter(train_loader))
    print(f"batch['audio'].shape: {batch['audio'].shape} Expecting [B, 1, T] B=batch size, 1=mono, T=number of samples")  # Expecting [B, 1, T] B=batch size, 1=mono, T=number of samples


    while global_step < max_steps:  
        for batch in train_loader:
            audio_input = batch["audio"].to(device)  # [B, 1, T]

            # match Lightning behavior
            train_discriminator = global_step >= pretrain_mel_steps
            loss_disc = torch.tensor(0.0, device=device)

            # ==================================================
            # DISCRIMINATOR STEP
            # ==================================================
            if train_discriminator:
                opt_disc.zero_grad()

                with torch.no_grad():
                    out = model(audio_input)
                    audio_hat = out["audio"]

                audio_input_1d = audio_input.squeeze(1)   # [B, 1, T] → [B, T]
                audio_hat_1d  = audio_hat.squeeze(1)    # [B, 1, T] → [B, T]

                # DAC discriminator loss
                loss_dac = dac_loss.discriminator_loss(
                    audio_hat, audio_input
                )

                # MPD
                real_mp, gen_mp, _, _ = disc_mpd(
                    y=audio_input_1d, y_hat=audio_hat_1d
                )
                loss_mp, loss_mp_real, _ = disc_loss_fn(
                    disc_real_outputs=real_mp,
                    disc_generated_outputs=gen_mp
                )
                loss_mp = loss_mp / len(loss_mp_real)

                # MRD
                real_mrd, gen_mrd, _, _ = disc_mrd(
                    y=audio_input_1d, y_hat=audio_hat_1d
                )
                loss_mrd, loss_mrd_real, _ = disc_loss_fn(
                    disc_real_outputs=real_mrd,
                    disc_generated_outputs=gen_mrd
                )
                loss_mrd = loss_mrd / len(loss_mrd_real)

                # total discriminator loss
                loss_disc = loss_mp + mrd_loss_coeff * loss_mrd + loss_dac

                loss_disc.backward()
                opt_disc.step()

            # ==================================================
            # GENERATOR STEP
            # ==================================================
            opt_gen.zero_grad()

            out = model(audio_input, bandwidth=6.6)
            audio_hat = out["audio"]
            commit_loss = out["commit_loss"]

            audio_input_1d = audio_input.squeeze(1)
            audio_hat_1d = audio_hat.squeeze(1)

            if train_discriminator:
                # DAC generator loss
                loss_dac_1, loss_dac_2 = dac_loss.generator_loss(
                    audio_hat,
                    audio_input
                )

                # MPD
                _, gen_mp, fmap_rs_mp, fmap_gs_mp = disc_mpd(
                    y=audio_input_1d, y_hat=audio_hat_1d
                )

                loss_gen_mp, list_loss_gen_mp = gen_loss_fn(gen_mp)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)

                loss_fm_mp = feat_match_loss_fn(
                    fmap_r=fmap_rs_mp,
                    fmap_g=fmap_gs_mp
                ) / len(fmap_rs_mp)

                # MRD
                _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = disc_mrd(
                    y=audio_input_1d, y_hat=audio_hat_1d
                )

                loss_gen_mrd, list_loss_gen_mrd = gen_loss_fn(gen_mrd)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)

                loss_fm_mrd = feat_match_loss_fn(
                    fmap_r=fmap_rs_mrd,
                    fmap_g=fmap_gs_mrd
                ) / len(fmap_rs_mrd)

            else:
                # pretraining phase
                loss_gen_mp = 0
                loss_gen_mrd = 0
                loss_fm_mp = 0
                loss_fm_mrd = 0
                loss_dac_1 = 0
                loss_dac_2 = 0


            # Mel loss
            mel_loss = mel_loss_fn(audio_hat, audio_input)

            # total generator loss
            loss_gen = (
                loss_gen_mp
                + mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + mrd_loss_coeff * loss_fm_mrd
                + mel_loss_coeff * mel_loss
                + 1000 * commit_loss
                + loss_dac_1
                + loss_dac_2
            )

            loss_gen.backward()
            opt_gen.step()

            # ==================================================
            # LOGGING & CHECKPOINTS
            # ==================================================
            if global_step % 10 == 0:
                print(
                    f"[{global_step}] "
                    f"G: {loss_gen.item():.4f} | "
                    f"D: {loss_disc.item():.4f} | "
                    f"Mel: {mel_loss.item():.4f} | "
                    f"Commit: {commit_loss.item():.6f}"
                )
                writer.add_scalar("loss/train_gen", loss_gen.item(), global_step)
                writer.add_scalar("loss/train_disc", loss_disc.item(), global_step)
                writer.add_scalar("loss/mel", mel_loss.item(), global_step)
                writer.add_scalar("loss/commit", commit_loss.item(), global_step)
                writer.add_scalar("loss/gen_mp", loss_gen_mp, global_step)
                writer.add_scalar("loss/gen_mrd", loss_gen_mrd, global_step)
            if global_step % 200 == 0:    
                writer.flush()

            if global_step != 0 and global_step % val_every == 0:
                val_loss, val_sample = validate(model, discriminators, val_loader, mel_loss_fn, device)
                writer.add_scalar("loss/val_mel", val_loss, global_step)
                writer.flush()
                print(f"[{global_step}] Val mel: {val_loss:.4f}")
                torchaudio.save(
                    str(val_samples_dir / f"val_{global_step}.wav"),
                    val_sample,
                    24000,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = {
                        "model": model.state_dict(),
                        "disc_mpd": disc_mpd.state_dict(),
                        "disc_mrd": disc_mrd.state_dict(),
                        "disc_dac": disc_dac.state_dict(),
                        "opt_gen": opt_gen.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "step": global_step,
                        "best_val_loss": best_val_loss,
                    }
                    torch.save(best_checkpoint, str(checkpoint_dir / "checkpoint_best.pt"))
                    print(f"✅ New best validation checkpoint: {best_val_loss:.4f}")

            if global_step != 0 and global_step % save_every == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "disc_mpd": disc_mpd.state_dict(),
                    "disc_mrd": disc_mrd.state_dict(),
                    "disc_dac": disc_dac.state_dict(),
                    "opt_gen": opt_gen.state_dict(),
                    "opt_disc": opt_disc.state_dict(),
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, str(checkpoint_dir / f"checkpoint_{global_step}.pt"))
                torch.save(checkpoint, str(checkpoint_dir / "checkpoint_latest.pt"))
                
                all_ckpts = sorted(
                    [p for p in checkpoint_dir.glob("checkpoint_*.pt") if p.name not in {"checkpoint_latest.pt", "checkpoint_best.pt"}],
                    key=os.path.getmtime,
                )

                if len(all_ckpts) > max_checkpoints:
                    for ck in all_ckpts[:-max_checkpoints]:
                        ck.unlink()

                print(f"✅ Saved checkpoint at step {global_step}")

            if global_step != 0 and global_step % sample_every == 0:
                torchaudio.save(
                    str(samples_dir / f"sample_{global_step}.wav"),
                    audio_hat[0].detach().cpu(),
                    24000,
                )

            global_step += 1

            # ==================================================
            # PROGRESS BAR UPDATE
            # ==================================================
            pbar.update(1)

            if global_step % 100 == 0:
                pbar.set_description(
                    f"G:{loss_gen.item():.2f} D:{loss_disc.item():.2f}"
                )
    writer.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(Path(args.config_path))
    main(config)