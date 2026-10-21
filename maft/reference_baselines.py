"""Reference method equations, adapted to the shared frozen CLIP features.

CoOp: KaiyangZhou/CoOp, trainers/coop.py (shared context, class token at end).
LaBo: YueYANG1996/LaBo, models/asso_opt/asso_opt.py (softmax association).
Tip-Adapter: gaopengcuhk/Tip-Adapter, main.py (trainable cache keys).
These are explicit ports, not claims to reproduce original published numbers.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CoOp(nn.Module):
    def __init__(self, clip_model, tokenizer, classes, n_ctx=16):
        super().__init__()
        self.clip = clip_model
        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        prefix = ' '.join(['X'] * n_ctx)
        tokens = tokenizer([prefix + ' ' + name.replace('_', ' ') + '.' for name in classes])
        tokens = tokens.to(next(clip_model.parameters()).device)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(tokens)
        self.register_buffer('prefix', embeddings[:, :1])
        self.register_buffer('suffix', embeddings[:, 1 + n_ctx:])
        self.register_buffer('tokens', tokens)
        self.ctx = nn.Parameter(torch.empty(n_ctx, embeddings.shape[-1], device=tokens.device))
        nn.init.normal_(self.ctx, std=.02)
        self.n_cls = len(classes)
        # Causal attention: padding after every EOT cannot affect any pooled EOT.
        self.sequence_length = int(tokens.argmax(-1).max().item()) + 1

    def text_features(self, truncate=True):
        length = self.sequence_length if truncate else self.tokens.shape[1]
        prompts = torch.cat([self.prefix, self.ctx[None].expand(self.n_cls, -1, -1), self.suffix], dim=1)
        x = prompts[:, :length] + self.clip.positional_embedding[:length]
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask[:length, :length])
        x = self.clip.ln_final(x)
        x = x[torch.arange(self.n_cls, device=x.device), self.tokens.argmax(-1)] @ self.clip.text_projection
        return F.normalize(x.float(), dim=-1)

    def forward(self, image_features):
        return self.clip.logit_scale.exp() * image_features @ self.text_features().T


class LaBo(nn.Module):
    def __init__(self, owners, n_classes):
        super().__init__()
        initial = torch.zeros(n_classes, len(owners), device=owners.device)
        initial.scatter_(0, owners.long().view(1, -1), 1.)
        self.asso_mat = nn.Parameter(initial)

    def forward(self, concept_scores):
        return 100 * (concept_scores @ self.asso_mat.softmax(dim=-1).T)


class TipAdapterF(nn.Module):
    def __init__(self, keys, labels, text_features, alpha=10., beta=1.):
        super().__init__()
        self.keys = nn.Parameter(keys.clone())
        self.register_buffer('values', F.one_hot(labels, len(text_features)).float())
        self.register_buffer('text_features', text_features)
        self.alpha, self.beta = alpha, beta

    def forward(self, features, alpha=None, beta=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        affinity = features @ self.keys.T
        cache_logits = (beta * (affinity - 1)).exp() @ self.values
        return 100 * features @ self.text_features.T + alpha * cache_logits
