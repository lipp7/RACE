"""Public RACE factory, including the paper's component/design ablations."""
from maft.spl_experiments import build, objective, forward_features


def build_model(metadata, config=None):
    """Construct RACE from class/concept text embeddings without image labels.

    metadata contains unit class_text [C,D] and concept_text [K,D]. A forward
    pass takes views [B,10,D] and returns logits plus the evidence intermediates.
    The internal parameter names are retained for checkpoint compatibility.
    """
    return build(metadata, dict(config or {}))


__all__ = ["build_model", "objective", "forward_features"]
