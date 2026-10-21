"""Controlled ports of ProLIP (WACV 2026) and ProKeR (CVPR 2025)."""
import torch
from torch import nn
from torch.nn import functional as F


class ProLIPProjection(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.vit_proj = nn.Parameter(projection.clone())
        self.register_buffer('initial', projection.clone())

    def forward(self, before, text):
        return 100. * F.normalize(before @ self.vit_proj, dim=-1) @ text.T

    def regularizer(self):
        # Author code uses SUM, not mean, of the projection displacement.
        return (self.vit_proj-self.initial).square().sum()


def rbf_kernel(x, support, beta):
    # Literal author RBF_Kernel; do not multiply zero-shot cosine by 100.
    return (-beta * (1.-x.float() @ support.float().T)).exp()


def proker_fit(support, labels, text, beta, lmbda):
    residual = F.one_hot(labels, num_classes=len(text)).float() - support @ text.T
    # Full-shot systems are much less well conditioned than the author's small
    # few-shot systems. Keep the literal FP32 kernel but solve it in FP64.
    matrix = rbf_kernel(support, support, beta).double() / lmbda
    matrix.diagonal().add_(1.)
    # SPD Cholesky solves the identical system used by torch.linalg.solve in the official code.
    factor = torch.linalg.cholesky(matrix)
    return torch.cholesky_solve(residual.double(), factor)


def proker_logits(x, support, text, coefficients, beta):
    return (x @ text.T).double() + rbf_kernel(x, support, beta).double() @ coefficients
