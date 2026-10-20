"""Model definitions shared by the Gradio inference application.

Architectures and layer shapes match the checkpoints produced by notebooks 02-04.
"""
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision.models import resnet18, vit_b_16

Architecture = Literal["custom_cnn", "resnet18", "vit_b16"]


class ConvBlock(nn.Module):
    """Convolution block matching notebook 02."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class CustomBrainMRICNN(nn.Module):
    """CNN architecture matching the model saved in notebook 02."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(architecture: Architecture | str, num_classes: int):
    """Build an uninitialized model with a head for ``num_classes`` labels.

    No pretrained weights are downloaded here; app inference loads the fine-tuned
    state dict saved by the relevant project notebook.
    """
    if num_classes < 2:
        raise ValueError(f"Expected at least two classes, received {num_classes}.")

    if architecture == "custom_cnn":
        return CustomBrainMRICNN(num_classes=num_classes)
    if architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "vit_b16":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(
        f"Unsupported architecture {architecture!r}. Choose custom_cnn, resnet18, or vit_b16."
    )


def gradcam_target_layer(model, architecture: Architecture | str):
    """Return a target module for the project's Grad-CAM implementation."""
    if architecture == "custom_cnn":
        return model.features[-1].block[5]
    if architecture == "resnet18":
        return model.layer4[-1]
    if architecture == "vit_b16":
        # The last encoder block outputs [batch, tokens, embedding_dim]. The
        # custom Grad-CAM routine below handles patch tokens separately.
        return model.encoder.layers[-1]
    raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture!r}")
