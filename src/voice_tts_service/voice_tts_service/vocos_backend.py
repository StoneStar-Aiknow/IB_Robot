"""Pinned Vocos implementation used by the verified ZipVoice checkpoint.

This is the inference subset of the 310P delivery's Vocos 0.1.0 source.  It
is project code, not model-bundle content, so its version and review history
are managed together with the adapter. It is adapted from
https://github.com/gemelo-ai/vocos (MIT License).
"""

from __future__ import annotations

import torch
from torch import nn

VOCOS_SOURCE_VERSION = "0.1.0-zipvoice-310p-delivery"
VOCOS_SOURCE_SHA256 = {
    "__init__.py": "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed",
    "heads.py": "af152896ca29255c6242a5156f18ca44e3553ab89b7921ee9c7dba32e0c3b74e",
    "models.py": "7031a1e223678cfeb522046f7e179e3ff9efd70aa32f969a4234880b59be1b71",
    "modules.py": "1132b0081333cc4abe1c36b2bd709185ceaa243519ea8903bec7ab08e67d5d64",
    "spectral_ops.py": "5577771daae83a9ffc6df3f1a3ce7f3f9c2543562c8ca3f5aff740ee2b515682",
}


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = False
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)
        x = self.norm(x).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same") -> None:
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("padding must be 'center' or 'same'")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        pad = (self.win_length - self.hop_length) // 2
        if spec.dim() != 3:
            raise ValueError("ISTFT expects a 3D complex spectrogram")
        _, _, frames = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        output_size = (1, (frames - 1) * self.hop_length + self.win_length)
        folded = torch.nn.functional.fold(
            ifft, output_size=output_size, kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        )[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, frames, -1).transpose(1, 2)
        envelope = torch.nn.functional.fold(
            window_sq, output_size=output_size, kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        ).squeeze()[pad:-pad]
        return folded / envelope


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same") -> None:
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        magnitude, phase = x.chunk(2, dim=1)
        magnitude = torch.clip(torch.exp(magnitude), max=1e2)
        spec = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        return self.istft(spec)


class ZipVoiceVocos(nn.Module):
    """Vocos decoder shape used by the fixed ZipVoice checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)
        self.head = ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(features))
