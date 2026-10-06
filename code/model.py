"""Backbone, worker layer, and evidential loss extracted from run/model.py."""

import torch
from torch import nn
from torch.nn import functional as F


class DenseNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CrowdLayerA(nn.Module):
    def __init__(self, backbone, num_workers, num_classes):
        super().__init__()
        self.backbone = backbone
        self.crowdlayer = nn.Parameter(
            torch.eye(num_classes).repeat(num_workers, 1, 1)
        )

    def forward(self, workers, x):
        evidence = self.backbone(x)
        alpha1 = evidence + 1
        # Keep the original worker-score transform; the loss applies ReLU + 1.
        alpha2 = torch.matmul(
            evidence.unsqueeze(-2), self.crowdlayer[workers]
        ).squeeze(-2)
        return alpha1, alpha2


def kl_divergence(alpha):
    ones = torch.ones_like(alpha[:1])
    strength = alpha.sum(dim=1, keepdim=True)
    first_term = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones) * (torch.digamma(alpha) - torch.digamma(strength))
    ).sum(dim=1, keepdim=True)
    return first_term + second_term


def edl_digamma_loss(output, labels, step, annealing_step=5):
    alpha = F.relu(output) + 1
    target = F.one_hot(labels, num_classes=output.shape[-1]).to(alpha.dtype)
    strength = alpha.sum(dim=1, keepdim=True)
    fit = (target * (torch.digamma(strength) - torch.digamma(alpha))).sum(
        dim=1, keepdim=True
    )
    kl_alpha = (alpha - 1) * (1 - target) + 1
    return (fit + min(1.0, step / annealing_step) * kl_divergence(kl_alpha)).mean()
