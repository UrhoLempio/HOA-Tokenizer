from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from utils import STFTParams, WNConv1d, WNConv2d

# DAC DISCRIMINATOR

class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 24000):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        # x = AudioSignal(x, self.sample_rate)
        # x.resample(self.sample_rate // self.rate)
        # x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 24000,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 24000
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        self.n_fft = window_length

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        # x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        # x = torch.view_as_real(x.stft())

        # x.squeeze(0).stft(n_fft=1024,win_length=1024,return_complex=True).size()
        # breakpoint()
        if x.size(0)==1:
            # x = torch.view_as_real(x.squeeze(0).stft(n_fft=self.window_length,return_complex=True).unsqueeze(0))
            x = torch.view_as_real(x.squeeze(0).stft(n_fft=self.n_fft,return_complex=True).unsqueeze(0))
        else:
            # x = torch.view_as_real(x.squeeze(1).stft(n_fft=self.window_length,return_complex=True).unsqueeze(1))
            x = torch.view_as_real(x.squeeze(1).stft(n_fft=self.n_fft,return_complex=True).unsqueeze(1))
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


# class DACDiscriminator(ml.BaseModel):
class DACDiscriminator(nn.Module):
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 24000,
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 24000
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


# DISCRIMINATORS


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator module adapted from https://github.com/jik876/hifi-gan.
    Additionally, it allows incorporating conditional information with a learned embeddings table.

    Args:
        periods (tuple[int]): Tuple of periods for each discriminator.
        num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
            Defaults to None.
    """

    def __init__(self, periods: Tuple[int] = (2, 3, 5, 7, 11), num_embeddings: int = None):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(period=p, num_embeddings=num_embeddings) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
        num_embeddings: int = None,
    ):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0))),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=1024)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.unsqueeze(1)
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int]] = ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        num_embeddings: int = None,
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/mindslab-ai/univnet.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            resolutions (tuple[tuple[int, int, int]]): Tuple of resolutions for each discriminator.
                Each resolution should be a tuple of (n_fft, hop_length, win_length).
            num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
                Defaults to None.
        """
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=r, num_embeddings=num_embeddings) for r in resolutions]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int],
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram


if __name__ == "__main__":
    disc = DACDiscriminator()
    x = torch.zeros(1, 1, 24000)
    results = disc(x)
    breakpoint()
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print("00")
