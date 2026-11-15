"""A compact DCGAN generator and discriminator for square images >= 16px."""

from __future__ import annotations

import torch.nn as nn


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, image_size: int = 64, channels: int = 64):
        super().__init__()
        if image_size < 16 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 16")
        stages = image_size.bit_length() - 3  # number of doublings from 4px
        start_channels = channels * min(8, 2 ** (stages - 1))
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, start_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(start_channels), nn.ReLU(True),
        ]
        current = start_channels
        for stage in range(stages - 1):
            nxt = max(channels, current // 2)
            layers.extend([nn.ConvTranspose2d(current, nxt, 4, 2, 1, bias=False), nn.BatchNorm2d(nxt), nn.ReLU(True)])
            current = nxt
        layers.extend([nn.ConvTranspose2d(current, 3, 4, 2, 1, bias=False), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 64, channels: int = 64):
        super().__init__()
        if image_size < 16 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 16")
        stages = image_size.bit_length() - 3
        layers: list[nn.Module] = [nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        current = channels
        for stage in range(stages - 1):
            nxt = min(channels * 8, current * 2)
            layers.extend([nn.Conv2d(current, nxt, 4, 2, 1, bias=False), nn.BatchNorm2d(nxt), nn.LeakyReLU(0.2, inplace=True)])
            current = nxt
        layers.append(nn.Conv2d(current, 1, 4, 1, 0))
        self.net = nn.Sequential(*layers)

    def forward(self, image):
        return self.net(image).view(-1)
