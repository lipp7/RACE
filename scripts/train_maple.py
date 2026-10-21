"""Matched ten-view PromptSRC/MaPLe using pinned official prompt modules.

The authors' CLIP/prompt classes are compiled without Dassl's trainer wrapper.
All changes to the experiment protocol are recorded in each config.json.
Run from the RACE project root with python -m scripts.train_maple.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'third_party/PromptSRC'
COMMIT = 'bb95c77b634d63488f2cad81ff4a72d53bdd06d5'
sys.path.insert(0, str(OFFICIAL))
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from scripts.extract_spl_features import dataset


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def official_namespace(method, state):
    """Load unchanged author definitions without importing optional Dassl."""
    def load_clip_to_cpu(cfg, zero_shot_model=False):
        if method == 'promptsrc':
            settings = cfg.TRAINER.PROMPTSRC
            design = dict(trainer='IVLP', vision_depth=0 if zero_shot_model else 9,
                          language_depth=0 if zero_shot_model else 9,
                          vision_ctx=0 if zero_shot_model else 4,
                          language_ctx=0 if zero_shot_model else 4)
        else:
            design = dict(trainer='MaPLe', vision_depth=0, language_depth=0,
                          vision_ctx=0, language_ctx=0, maple_length=2)
        model = clip.build_model(dict(state), design).float()
        # The only newly initialized weights must be prompt parameters.
        for name, value in state.items():
            if name in model.state_dict():
                assert torch.equal(value.float(), model.state_dict()[name].float()), name
        return model

    template_ns = {}
    exec(compile((OFFICIAL / 'trainers/imagenet_templates.py').read_text(), 'imagenet_templates.py', 'exec'), template_ns)
    namespace = dict(torch=torch, nn=nn, F=F, copy=copy, clip=clip,
                     _tokenizer=SimpleTokenizer(), load_clip_to_cpu=load_clip_to_cpu,
                     IMAGENET_TEMPLATES=template_ns['IMAGENET_TEMPLATES'])
    tree = ast.parse((OFFICIAL / f'trainers/{method}.py').read_text())
    names = {'TextEncoder', 'VLPromptLearner', 'MultiModalPromptLearner', '_get_clones'}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(OFFICIAL / f'trainers/{method}.py'), 'exec'), namespace)
    return namespace


class MatchedPromptModel(nn.Module):
    def __init__(self, method, classes, state):
        super().__init__()
        self.method = method
        cfg = NS(INPUT=NS(SIZE=[224]), TRAINER=NS(
            PROMPTSRC=NS(PROMPT_DEPTH_TEXT=9, PROMPT_DEPTH_VISION=9, N_CTX_TEXT=4,
                         N_CTX_VISION=4, CTX_INIT='a photo of a'),
            MAPLE=NS(N_CTX=2, CTX_INIT='a photo of a', PROMPT_DEPTH=9)))
        ns = official_namespace(method, state)
        base = ns['load_clip_to_cpu'](cfg)
        learner = ns['VLPromptLearner' if method == 'promptsrc' else 'MultiModalPromptLearner']
        self.prompt_learner = learner(cfg, classes, base)
        self.image_encoder = base.visual
        self.text_encoder = ns['TextEncoder'](base)
        self.logit_scale = base.logit_scale
        self.tokens = self.prompt_learner.tokenized_prompts
        # Teacher encoders are frozen, including the one nested in prompt_learner.
        for name, param in self.named_parameters():
            param.requires_grad_(('prompt_learner' in name and 'ZS_image_encoder' not in name) or 'VPT' in name)
        self.float().cuda()
        if method == 'promptsrc':
            self.register_buffer('teacher_text', F.normalize(self.prompt_learner.fixed_embeddings.float(), dim=-1))
        self.trainable = {name: param for name, param in self.named_parameters() if param.requires_grad}
        assert self.trainable and all('prompt_learner' in name or 'VPT' in name for name in self.trainable)

    def features(self, images):
        batch, views = images.shape[:2]
        assert views == 10
        x = images.flatten(0, 1)
        if self.method == 'promptsrc':
            prompts = self.prompt_learner()
            text = self.text_encoder(prompts, self.tokens)
            image = self.image_encoder(x)
        else:
            prompts, shared, deep_text, deep_visual = self.prompt_learner()
            text = self.text_encoder(prompts, self.tokens, deep_text)
            image = self.image_encoder(x, shared, deep_visual)
        per_view = F.normalize(image.float(), dim=-1).reshape(batch, views, -1)
        pooled = F.normalize(per_view.mean(1), dim=-1)
        return pooled, F.normalize(text.float(), dim=-1), per_view

    def forward(self, images, labels=None):
        image, text, per_view = self.features(images)
        scale = self.logit_scale.exp()
        logits = scale * image @ text.T
        if labels is None:
            return logits
        ce = F.cross_entropy(logits, labels)
        if self.method == 'maple':
            return ce, {'ce': float(ce.detach())}
        with torch.no_grad():
            teacher_views = F.normalize(self.prompt_learner.ZS_image_encoder(images.flatten(0, 1)).float(), dim=-1).reshape(len(images), 10, -1)
            teacher_image = F.normalize(teacher_views.mean(1), dim=-1)
            teacher_logits = scale * teacher_image @ self.teacher_text.T
        # Preserve the official L1-text (25), L1-image (10) and KL normalization.
        # L1-image acts on the per-view features; class prediction uses their mean.
        text_loss = 25 * F.l1_loss(text, self.teacher_text)
        image_loss = 10 * F.l1_loss(per_view, teacher_views)
        kl = F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(teacher_logits, dim=1),
                      reduction='sum', log_target=True) / logits.numel()
        loss = ce + text_loss + image_loss + kl
        return loss, {k: float(v.detach()) for k, v in dict(ce=ce, text=text_loss, image=image_loss, kl=kl).items()}

    def prompt_state(self):
        return {name: value.detach().float().cpu().clone() for name, value in self.trainable.items()}

    @torch.no_grad()
    def load_prompt_state(self, state):
        for name, value in self.trainable.items():
            value.copy_(state[name])


@torch.no_grad()
def evaluate(model, loader, predictions=False):
    model.eval()
    correct = count = 0
    all_logits, all_y = [], []
    for images, y in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        assert torch.isfinite(logits).all()
        correct += (logits.cpu().argmax(-1) == y).sum().item()
        count += len(y)
        if predictions:
            all_logits.append(logits.cpu())
            all_y.append(y)
    return correct / count, count, (torch.cat(all_logits), torch.cat(all_y)) if predictions else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['maple'], default='maple')
    parser.add_argument('--features', help='Relative ten-view cache directory')
    parser.add_argument('--state', default='cache/openai_b32_state.pt')
    parser.add_argument('--dataset', choices=['cub', 'pets', 'flowers', 'cars', 'aircraft'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    from race.io import project_root, relative_path
    from scripts.train import stratified_split
    project_root()
    relative_path(args.output)
    features = relative_path(args.features or f'features/spl_fresh32_{args.dataset}')
    meta = torch.load(features / 'metadata.pt', weights_only=False)
    source_labels = torch.load(features / 'train.pt', weights_only=True)['label']
    train_ids, val_ids = stratified_split(source_labels, args.seed)
    split = dict(train_ids=train_ids, val_ids=val_ids)
    split_path = output / 'split.pt'
    torch.save(split, split_path)
    reference = dict(run=str(features), train_n=len(train_ids), val_n=len(val_ids),
                     split_sha256=sha(split_path))
    # Reuse precisely the transform that generated the reference ten-view cache.
    from open_clip.transform import image_transform
    preprocess = image_transform(224, is_train=False, mean=(.48145466, .4578275, .40821073),
                                 std=(.26862954, .26130258, .27577711), interpolation='bicubic')
    source = dataset(args.dataset, 'train', preprocess)
    cached = torch.load(features / 'train.pt', weights_only=True)
    # Verify ordering for every example, without decoding images.
    labels = torch.tensor([int(item[1]) for item in source.items])
    assert torch.equal(labels, cached['label'])
    assert len(split['train_ids']) == reference['train_n'] and len(split['val_ids']) == reference['val_n']
    reference_first_views = cached['views'][0].float().clone()
    del cached
    train = Subset(source, split['train_ids'].tolist())
    val = Subset(source, split['val_ids'].tolist())
    if args.smoke:
        train = Subset(train, list(range(8)))
        val = Subset(val, list(range(8)))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=4, shuffle=True, generator=generator,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    state_path = relative_path(args.state)
    if not state_path.exists():
        from race.prepare_maple import prepare_state
        prepare_state(state_path)
    state = torch.load(state_path, weights_only=True)
    model = MatchedPromptModel(args.method, meta['classes'], state)
    # Check pixel transforms, crop order, frozen encoder weights and reference cache.
    if args.method == 'promptsrc':
        audit_encoder = model.prompt_learner.ZS_image_encoder
    else:
        audit_model = clip.build_model(dict(state), dict(trainer='IVLP', vision_depth=0,
                        language_depth=0, vision_ctx=0, language_ctx=0)).float().cuda()
        audit_encoder = audit_model.visual
    with torch.no_grad():
        audit_input, _ = source[0]
        audit_views = F.normalize(audit_encoder(audit_input.cuda()).float(), dim=-1).cpu()
        cosine = F.cosine_similarity(audit_views, reference_first_views, dim=-1)
        max_error = float((audit_views - reference_first_views).abs().max())
        assert float(cosine.min()) > .99999 and max_error < .002, (cosine, max_error)
    atomic_json(output / 'input_audit.json', dict(actual_view_shape=list(audit_input.shape),
                reference_cache_min_cosine=float(cosine.min()), reference_cache_max_abs_error=max_error,
                raw_label_order_matches_cache=True, reference_split_sha256=sha(split_path)))
    if args.method == 'maple':
        del audit_model
    del audit_encoder
    epochs = 20 if args.method == 'promptsrc' else 5
    lr = .0025 if args.method == 'promptsrc' else .0035
    config = dict(**vars(args), backbone='ViT-B/32', pretrained='OpenAI', view_count=10,
                  aggregation='L2 normalize each view, arithmetic mean, L2 normalize mean',
                  view_geometry='one global + 3x3 overlapping half-size crops',
                  supervision='full training split, closed-set', online_augmentation=False,
                  official_repo='https://github.com/muzairkhattak/PromptSRC', official_commit=COMMIT,
                  official_trainer_sha256=sha(OFFICIAL / f'trainers/{args.method}.py'),
                  clip_model_sha256=sha(OFFICIAL / 'clip/model.py'),
                  wrapper_sha256=sha(Path(__file__)), pretrained_state_sha256=sha(state_path),
                  reference_cache=reference['run'], reference_split_sha256=sha(split_path),
                  train_n=len(train), val_n=len(val), batch_size=4, epochs=epochs, lr=lr,
                  optimizer='SGD', momentum=.9, weight_decay=.0005, warmup_lr=1e-5,
                  warmup_epochs=1, precision='FP32 parameters with BF16 autocast',
                  prompt_depth=9, context_tokens=4 if args.method == 'promptsrc' else 2,
                  selection='earliest validation-best checkpoint; PromptSRC uses normalized prefix Gaussian prompt ensemble',
                  official_changes=['ViT-B/32 in place of ViT-B/16', 'full supervision and shared stratified splits',
                                    'deterministic matched ten-view inputs and mean aggregation',
                                    'BF16 autocast with FP32 parameters', 'validation selection; no test access during training'],
                  trainable_parameters={k: list(v.shape) for k, v in model.trainable.items()})
    if args.method == 'promptsrc':
        config.update(text_loss_weight=25, image_loss_weight=10, teacher_templates=80,
                      gaussian_mean=15, gaussian_std=1,
                      regularization='text L1, per-view image L1, KL on aggregated class logits')
        config['official_changes'].append('validate normalized prefix GPA checkpoints; final epoch equals official full Gaussian average')
    else:
        config['epoch_source']='original MaPLe config MAX_EPOCH=5; PromptSRC fork config says 2'
    if args.method == 'maple':
        original = ROOT / 'third_party/MaPLe'
        assert sha(original / 'trainers/maple.py') == sha(OFFICIAL / 'trainers/maple.py')
        config['original_repo'] = 'https://github.com/muzairkhattak/multimodal-prompt-learning'
        config['original_commit'] = '69bce21ae8eda80ad6187534b2dce09cf6c59e17'
        config['original_config_sha256'] = sha(original / 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml')
    atomic_json(output / 'config.json', config)
    del state
    optimizer = torch.optim.SGD(model.trainable.values(), lr=lr, momentum=.9, weight_decay=.0005)
    history, best, best_epoch = [], -1., None
    gaussian_average, gaussian_log_weight = None, -math.inf
    start_epoch = 1
    if (output / 'latest.pt').exists() and not args.smoke:
        saved = torch.load(output / 'latest.pt', weights_only=False)
        model.load_prompt_state(saved['raw_prompt'])
        optimizer.load_state_dict(saved['optimizer'])
        gaussian_average = saved['gaussian_average']
        gaussian_log_weight = saved['gaussian_log_weight']
        history, best, best_epoch = saved['history'], saved['best'], saved['best_epoch']
        start_epoch = saved['epoch'] + 1
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
        print('RESUME', start_epoch, flush=True)
    epoch_range = range(start_epoch, 2 if args.smoke else epochs + 1)
    for epoch in epoch_range:
        rate = 1e-5 if epoch == 1 else lr * .5 * (1 + math.cos(math.pi * (epoch - 2) / (epochs - 1)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        losses = []
        for step, (images, y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, parts = model(images.cuda(non_blocking=True), y.cuda(non_blocking=True))
            assert torch.isfinite(loss), parts
            loss.backward()
            if epoch == 1 and step == 0:
                grads = {name: float(p.grad.float().norm()) if p.grad is not None else None for name, p in model.trainable.items()}
                assert all(v is not None and math.isfinite(v) for v in grads.values()), grads
                assert any(v > 0 for k, v in grads.items() if 'image' in k or 'proj' in k)
                assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
                atomic_json(output / 'gradient_audit.json', dict(trainable_gradient_norms=grads,
                            frozen_parameters_with_gradient=[name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None],
                            actual_image_shape=list(images.shape), loss=parts))
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, steps=len(train_loader), loss=losses[-1], seconds=time.monotonic()-started)), flush=True)
        raw = model.prompt_state()
        if args.method == 'promptsrc':
            log_weight = -.5 * ((epoch - 15) / 1.) ** 2
            new_log_weight = float(np.logaddexp(gaussian_log_weight, log_weight))
            fraction = math.exp(log_weight - new_log_weight)
            gaussian_average = raw if gaussian_average is None else {
                k: gaussian_average[k] * (1 - fraction) + v * fraction for k, v in raw.items()}
            gaussian_log_weight = new_log_weight
            model.load_prompt_state(gaussian_average)
        score, count, _ = evaluate(model, val_loader)
        entry = dict(epoch=epoch, lr=rate, loss=sum(losses)/len(losses), val_accuracy=score, val_n=count, seconds=time.monotonic()-started)
        history.append(entry)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.prompt_state(), output / 'best.pt')
        torch.save(dict(epoch=epoch, raw_prompt=raw, optimizer=optimizer.state_dict(),
                        gaussian_average=gaussian_average, gaussian_log_weight=gaussian_log_weight,
                        history=history, best=best, best_epoch=best_epoch,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        loader_rng=generator.get_state()), output / 'latest.pt')
        model.load_prompt_state(raw)
        atomic_json(output / 'progress.json', dict(status='training', **entry, best_val_accuracy=best, best_epoch=best_epoch))
        print(json.dumps(entry), flush=True)
    result = dict(method=args.method, dataset=args.dataset, seed=args.seed, view_count=10,
                  val_accuracy=best, best_epoch=best_epoch, history=history,
                  seconds=time.monotonic()-started, complete=not args.smoke)
    if args.smoke:
        atomic_json(output / 'smoke.json', result)
        print('SMOKE_OK', args.method, args.dataset, flush=True)
        return
    atomic_json(output / 'result.json', result)
    # Only instantiate/read the official test split after checkpoint selection.
    model.load_prompt_state(torch.load(output / 'best.pt', weights_only=True))
    test = dataset(args.dataset, 'test', preprocess)
    test_cached = torch.load(ROOT / f'features/spl_fresh32_{args.dataset}/test.pt', weights_only=True)
    assert torch.equal(torch.tensor([int(item[1]) for item in test.items]), test_cached['label'])
    assert len(test) == reference['test_n']
    del test_cached
    test_loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    score, count, (logits, target) = evaluate(model, test_loader, predictions=True)
    torch.save(dict(logits=logits, label=target), output / 'test_predictions.pt')
    evaluation = dict(test_accuracy=score, test_n=count, best_epoch=best_epoch,
                      checkpoint_sha256=sha(output / 'best.pt'), split_sha256=sha(split_path),
                      result_sha256=sha(output / 'result.json'), selection='validation only')
    atomic_json(output / 'evaluation_full.json', evaluation)
    atomic_json(output / 'progress.json', dict(status='complete', **evaluation, seconds=time.monotonic()-started))
    print(json.dumps(evaluation), flush=True)


if __name__ == '__main__':
    main()
