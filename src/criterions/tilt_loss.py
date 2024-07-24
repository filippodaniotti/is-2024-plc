import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal as signal
import torchaudio.functional as AF


def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, order, left_pad):
    data_ = data.cpu().numpy()
    data_ = np.pad(data_, ((0, 0), (left_pad, 0)), mode="edge")
    b, a = butter_lowpass(cutoff, data_.shape[1], order=order)
    y = signal.lfilter(b, a, data_)
    y = y[:, left_pad:]
    return torch.tensor(y).type_as(data)


def _spectrogram(
    wave, fft_size, hop_length, win_length, frequency_scale, sample_rate, **kwargs
):
    spec = (
        torch.stft(
            wave,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).type_as(wave),
            center=True,
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        .abs()
        .clamp(min=1e-7)
    )

    if frequency_scale == "mel":
        spec = spec.transpose(-1, -2) @ AF.melscale_fbanks(
            n_freqs=fft_size // 2 + 1,
            n_mels=kwargs.get("num_mel_bins", 128),
            sample_rate=sample_rate,
            f_min=kwargs.get("lower_edge_hertz", 0.0),
            f_max=kwargs.get("upper_edge_hertz", sample_rate / 2.0),
        ).type_as(wave)
        spec = spec.transpose(-1, -2)

    return spec


def _simple_reweight_mask(specs, **kwargs):
    frequency_bins = specs.shape[1]
    left_clamp_length = kwargs.get("left_clamp_length", 0)
    right_clamp_length = kwargs.get("right_clamp_length", 0)
    mask_length = frequency_bins - left_clamp_length - right_clamp_length
    left_clamp = torch.zeros((left_clamp_length,)).type_as(specs)
    right_clamp = torch.ones((right_clamp_length,)).type_as(specs)
    mask = torch.linspace(0.0, 1.0, steps=mask_length).type_as(specs)
    mask = torch.cat((left_clamp, mask, right_clamp), dim=0)
    mask = mask.unsqueeze(0).unsqueeze(2)
    return mask


def _fft_reweight_mask(specs, **kwargs):
    targets = kwargs["target"].detach()

    fft_size = (
        2 << torch.log2(torch.tensor(targets.shape[1])).floor().int().cpu().numpy()
    )

    spectrum = _spectrogram(
        targets,
        fft_size=fft_size,
        hop_length=targets.shape[1] + 1,
        win_length=targets.shape[1],
        frequency_scale="mel",
        sample_rate=kwargs.get("sample_rate"),
    ).squeeze()

    spectrum = 20 * torch.log10(spectrum) + 80.0
    spectrum = butter_lowpass_filter(
        spectrum,
        kwargs.get("cutoff", 10),
        kwargs.get("order", 5),
        kwargs.get("left_pad", 50),
    )
    spectrum = spectrum / spectrum.max(dim=1, keepdim=True).values
    spectrum = spectrum.unsqueeze(2)
    spectrum = 1 - spectrum
    return spectrum


reweight_strategies = {
    "linear": lambda specs, **kwargs: _simple_reweight_mask(specs, **kwargs),
    "quadratic": lambda specs, **kwargs: _simple_reweight_mask(specs, **kwargs).pow(2),
    "spectral": lambda specs, **kwargs: _fft_reweight_mask(specs, **kwargs),
    "none": lambda specs, **kwargs: torch.ones((specs.shape[1],)).type_as(specs),
}


class TiltLoss(nn.Module):
    def __init__(
        self,
        reweight_strategy: str,
        sample_rate: int,
        fft_size: int,
        hop_length: int,
        win_length: int,
        frequency_scale: str,
        lmbda: float,
        **kwargs
    ):
        super().__init__()
        self.reweight_strategy = reweight_strategies[reweight_strategy]
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.frequency_scale = frequency_scale
        self.lambda_ = lmbda

        self.kwargs = kwargs
        self.kwargs["sample_rate"] = sample_rate

    def forward(self, pred, target):
        pred_ = _spectrogram(
            pred,
            self.fft_size,
            self.hop_length,
            self.win_length,
            self.frequency_scale,
            self.sample_rate,
        )
        target_ = _spectrogram(
            target,
            self.fft_size,
            self.hop_length,
            self.win_length,
            self.frequency_scale,
            self.sample_rate,
        )

        diff = F.l1_loss(pred_, target_, reduction="none")
        diff = self._apply_reweighting(diff, target)
        # diff = self._apply_reweighting(diff, pred - target)

        return diff.mean()

    def _apply_reweighting(self, diff: torch.tensor, target: torch.tensor):
        reweight_mask = self.reweight_strategy(
            diff, **self.kwargs, target=target
        ).expand(-1, -1, diff.shape[2])
        return diff * reweight_mask


class TiltLossBuilder:
    def __init__(self):
        self._instance = None

    def __call__(
        self,
        reweight_strategy: str = "linear",
        sample_rate: int = 32000,
        fft_size: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        frequency_scale: str = "linear",
        lmbda: float = 1.0e3,
        **kwargs
    ):
        if self._instance is None:
            self._instance = TiltLoss(
                reweight_strategy=reweight_strategy,
                sample_rate=sample_rate,
                fft_size=fft_size,
                hop_length=hop_length,
                win_length=win_length,
                frequency_scale=frequency_scale,
                lmbda=lmbda,
                kwargs=kwargs,
            )
        return self._instance
