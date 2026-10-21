import unittest
import torch
from torch.nn import functional as F
from race.model import build_model, forward_features, objective
from maft.semantic_preservation import calibrate, preservation_metrics


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.meta = dict(class_text=torch.randn(5, 32), concept_text=torch.randn(12, 32))
        self.views = torch.randn(8, 10, 32)
        self.labels = torch.arange(8) % 5

    def test_forward_equations_and_losses(self):
        model = build_model(self.meta)
        out = model(self.views)
        rows = torch.arange(len(self.labels))
        masked = out["base_logits"].detach()[:, None].expand(-1, 5, -1).clone()
        masked.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
        rival = masked.argmax(-1)
        self.assertTrue(torch.equal(rival, out["rival_ids"]))
        other_gate = out["gate"].gather(1, rival[:, :, None].expand_as(out["gate"]))
        gap = out["gate"] - other_gate
        private = F.normalize((gap / .3).softmax(-1) @ out["local_views"], dim=-1)
        shared = F.normalize((torch.minimum(out["gate"], other_gate) / .3).softmax(-1) @ out["local_views"], dim=-1)
        eta = (gap / .5).sigmoid().mean(-1)
        other_anchor = out["anchor"].gather(1, rival[:, :, None].expand_as(out["anchor"]))
        utility = eta * ((private - shared) * F.normalize(out["anchor"] - other_anchor, dim=-1)).sum(-1)
        self.assertTrue(torch.allclose(utility, out["identity"], atol=1e-6))
        self.assertTrue(torch.allclose(out["logits"], out["base_logits"] + out["effective_scale"] * utility))
        y = self.labels
        j = rival[rows, y]
        hinge = lambda t: .1 * F.softplus((.01 - t) / .1)
        id_loss = hinge(utility[rows, y] - utility[rows, j]).mean()
        attack = hinge((private[rows, y] * (out["anchor"][rows, y] - out["anchor"][rows, j])).sum(-1)).mean()
        defense = ((shared[rows, y] * (out["anchor"][rows, y] - out["anchor"][rows, j])).sum(-1)).square().mean()
        expected = F.cross_entropy(out["logits"], y) + .3 * id_loss + .03 * attack + .03 * defense
        loss, _ = objective(out, y, {})
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        loss.backward()
        self.assertGreater(float(model.visual_adapter[-1].weight.grad.norm()), 0)
        self.assertGreater(float(model.conditioner[-1].weight.grad.norm()), 0)

    def test_training_only_preservation(self):
        model = build_model(self.meta)
        statistics = calibrate(self.views, torch.arange(6), {"concept": F.normalize(self.meta["concept_text"], dim=-1)}, device="cpu")
        with torch.no_grad():
            model.visual_adapter[-1].weight.normal_(std=.01)
        _, original, adapted = forward_features(model, self.views)
        actual = preservation_metrics(original, adapted, statistics)["concept"]
        matrix = statistics["concept"]["bank"] / statistics["concept"]["std"][:, None]
        expected = .5 * ((adapted - original) @ matrix.T).square().mean()
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(statistics["concept"]["std"].requires_grad)

    def test_base_and_no_subtraction(self):
        base = build_model(self.meta, {"mode": "ng"})
        out = base(self.views)
        self.assertTrue(torch.equal(out["logits"], out["base_logits"]))
        model = build_model(self.meta, {"evidence": "no_shared_residual"})
        out = model(self.views)
        other = out["anchor"].gather(1, out["rival_ids"][:, :, None].expand_as(out["anchor"]))
        expected = out["support_mass"] * (out["plus"] * F.normalize(out["anchor"] - other, dim=-1)).sum(-1)
        self.assertTrue(torch.allclose(expected, out["identity"]))


if __name__ == "__main__":
    unittest.main()
