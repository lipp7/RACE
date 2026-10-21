"""Training-only adapter constraints with a fixed concept or random measurement bank."""
import torch
from torch.nn import functional as F


def forward_with_adapter_features(model, views):
    captured = []
    def capture(module, args, delta):
        captured.append((args[0], delta))
    assert model.visual_adapter is not None and model.visual_adapter_scope == 'all'
    handle = model.visual_adapter.register_forward_hook(capture)
    try:
        out = model(views)
    finally:
        handle.remove()
    assert len(captured) == 1
    original, delta = captured[0]
    adapted = F.normalize(original + delta, dim=-1)
    return out, original.detach(), adapted


def banks(concept_text):
    concepts = F.normalize(concept_text.float(), dim=-1)
    generator = torch.Generator().manual_seed(20261015)
    random = F.normalize(torch.randn(concepts.shape, generator=generator), dim=-1)
    return dict(concept=concepts, random=random)


@torch.no_grad()
def calibrate(views, train_ids, measurements, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    statistics = {}
    for name, bank in measurements.items():
        bank = bank.to(device)
        sums = torch.zeros(len(bank), device=device)
        squares = torch.zeros_like(sums)
        count = 0
        for ids in train_ids.split(128):
            original = F.normalize(views[ids].to(device).float(), dim=-1)
            response = original @ bank.T
            sums += response.sum((0, 1))
            squares += response.square().sum((0, 1))
            count += response.shape[0] * response.shape[1]
        mu = sums / count
        std = (squares / count - mu.square()).clamp_min(0).sqrt().clamp_min(.03)
        statistics[name] = dict(bank=bank.cpu(), mean=mu.cpu(), std=std.cpu())
    return statistics


def preservation_metrics(original, adapted, statistics):
    metrics = dict(geometry=(1 - (original * adapted).sum(-1)).mean())
    for name, values in statistics.items():
        difference = (adapted - original) @ values['bank'].T
        # Half mean square standardized response change; std is train-only.
        metrics[name] = .5 * (difference / values['std']).square().mean()
    return metrics
