"""Train reference baselines without reading test data.

All use the shared ViT-B/32 OpenAI weights and existing stratified splits.
Cached global center-crop features replace online image augmentation. CoOp
still backpropagates through the real frozen text encoder into context tokens.
LaBo keeps the official all-shot Adam settings and maximum training budget.
Tip-Adapter-F selects its checkpoint on validation, correcting the official
script's use of test accuracy for checkpoint selection.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from maft.reference_baselines import CoOp, LaBo, TipAdapterF
from scripts.train import stratified_split
from maft.protocol import fewshot_ids


@torch.no_grad()
def accuracy(model, features, labels):
    # CoOp text vectors are shared across all validation images.
    if isinstance(model, CoOp):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            text = model.text_features()
        predicted = (features @ text.T).argmax(-1)
    else:
        predicted = torch.cat([model(x).argmax(-1) for x in features.split(256)])
    return (predicted == labels).float().mean().item()


def snapshot(model):
    if isinstance(model, CoOp):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            text = model.text_features()
        return {'ctx': model.ctx.detach().cpu(), 'text_features': text.detach().cpu()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def tip_search(model, features, labels):
    """Same 200x20 beta/alpha grid formula as the official implementation."""
    with torch.no_grad():
        affinity = torch.cat([x @ model.keys.T for x in features.split(256)])
        clip = torch.cat([100 * x @ model.text_features.T for x in features.split(256)])
        alphas = torch.tensor([i * (50 - .1) / 20 + .1 for i in range(20)], device=features.device)
        best_acc = accuracy(model, features, labels)
        best_beta, best_alpha = model.beta, model.alpha
        for i in range(200):
            beta = i * (50 - .1) / 200 + .1
            cache = torch.cat([(beta * (x - 1)).exp() @ model.values for x in affinity.split(256)])
            logits = clip[None] + alphas[:, None, None] * cache[None]
            if not torch.isfinite(logits).all():
                continue
            acc = (logits.argmax(-1) == labels[None]).float().mean(-1)
            value, index = acc.max(0)
            if value.item() > best_acc:
                best_acc = value.item()
                best_beta, best_alpha = beta, alphas[index].item()
        return {'alpha': best_alpha, 'beta': best_beta, 'val_accuracy': best_acc}


def linear_probe(train, y, val, vy, n_classes, output, max_iterations=500):
    """L2 logistic regression: CE + ||W||^2/(2*C*N), unpenalized bias."""
    trials = []
    best = -1
    for c in (1e-6, 1e-4, 1e-2, 1., 1e2, 1e4, 1e6):
        model = nn.Linear(train.shape[-1], n_classes).to(train.device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=max_iterations, tolerance_grad=1e-7,
                               tolerance_change=1e-9, line_search_fn='strong_wolfe')
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            opt.zero_grad()
            loss = F.cross_entropy(model(train), y) + model.weight.square().sum() / (2 * c * len(y))
            loss.backward()
            return loss
        opt.step(closure)
        score = accuracy(model, val, vy)
        entry = {'C': c, 'val_accuracy': score, 'closure_calls': calls,
                 'iterations': opt.state[model.weight]['n_iter']}
        trials.append(entry)
        print(json.dumps(entry), flush=True)
        if score > best:
            best = score
            torch.save(snapshot(model), output / 'best.pt')
            best_c = c
    return {'val_accuracy': best, 'best_C': best_c, 'history': trials, 'best_epoch': None}


def reference_features(data, mode='global'):
    if mode == 'global':
        value = data['image'].float()
    elif mode == 'viewmean':
        value = F.normalize(data['views'].float(), dim=-1).mean(1)
    elif mode == 'viewconcat':
        value = F.normalize(data['views'].float(), dim=-1).flatten(1)
    else:
        raise ValueError(mode)
    return F.normalize(value, dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=('cub', 'pets', 'flowers', 'cars', 'aircraft'))
    p.add_argument('--method', required=True, choices=('coop', 'labo', 'labo_raw', 'tip_f', 'tip', 'linear'))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--shots',type=int,default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--linear-max-iterations', type=int, default=500)
    p.add_argument('--features', help='Override cached feature directory for encoder controls')
    p.add_argument('--feature-mode', choices=('global', 'viewmean', 'viewconcat'), default='global')
    args = p.parse_args()
    if args.feature_mode != 'global' and args.method not in ('linear', 'coop'):
        raise ValueError('Multi-view controls currently support linear and CoOp')
    if args.feature_mode == 'viewconcat' and args.method != 'linear':
        raise ValueError('Concatenation is a linear probe control')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    root = Path(args.features or f'features/spl_fresh32_{args.dataset}')
    meta = torch.load(root / 'metadata.pt', weights_only=False)
    data = torch.load(root / 'train.pt', weights_only=True)
    features = (torch.load(f'features/labo_raw_{args.dataset}/raw_train.pt', weights_only=True)['image'].cuda()
                if args.method == 'labo_raw' else reference_features(data, args.feature_mode).cuda())
    labels = data['label'].cuda()
    train_ids, val_ids = stratified_split(data['label'], args.seed)
    train_ids = fewshot_ids(data['label'],train_ids,args.shots,args.seed)
    train, val, y, vy = features[train_ids], features[val_ids], labels[train_ids], labels[val_ids]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'backbone': meta['model'], 'pretrained': meta['pretrained'],
              'view': args.feature_mode, 'online_augmentation': False,
              'selection': 'validation only; earliest checkpoint wins ties', 'batch_size': None,
              'train_count': len(train_ids), 'val_count': len(val_ids)}
    torch.save({'train_ids': train_ids, 'val_ids': val_ids}, output / 'split.pt')
    if args.method == 'linear':
        result = linear_probe(train, y, val, vy, len(meta['classes']), output, args.linear_max_iterations)
        config.update({'optimizer': 'LBFGS', 'max_iterations': args.linear_max_iterations, 'C_grid': [1e-6, 1e-4, 1e-2, 1., 1e2, 1e4, 1e6]})
    else:
        if args.method == 'coop':
            import open_clip
            clip = open_clip.create_model(meta['model'], pretrained=meta['pretrained'], cache_dir='cache/open_clip').eval().cuda()
            model = CoOp(clip, open_clip.get_tokenizer(meta['model']), meta['classes'])
            epochs, batch_size, lr = 200, 128, .002
            optimizer = torch.optim.SGD([model.ctx], lr=lr, momentum=.9, weight_decay=.0005)
            config.update({'context_tokens': 16, 'shared_context': True, 'class_token_position': 'end',
                           'optimizer': 'SGD', 'momentum': .9, 'weight_decay': .0005,
                           'warmup_epochs': 1, 'warmup_lr': 1e-5, 'precision': 'bf16 autocast',
                           'official_changes': ['cached center crop, no online augmentation', 'batch size 128 instead of 32']})
        elif args.method.startswith('labo'):
            tag=f'fs{args.shots}_' if args.shots else ''
            selected_path = Path(f"features/labo_{'raw_' if args.method == 'labo_raw' else ''}{args.dataset}/selected_{tag}s{args.seed}.pt")
            concepts = torch.load(selected_path, weights_only=True)
            assert torch.equal(train_ids, concepts['train_ids']) and torch.equal(val_ids, concepts['val_ids'])
            text = concepts['text'].cuda()
            train, val = train @ text.T, val @ text.T
            model = LaBo(concepts['owners'].cuda(), len(meta['classes']))
            epochs = 5000 if args.dataset == 'cub' else 10000
            batch_size = 512 if args.dataset == 'cub' else 256
            lr = 1e-5 if args.dataset == 'flowers' else 5e-5
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            config.update({'optimizer': 'Adam', 'concept_path': str(selected_path), 'concept_count': len(text),
                           'asso_act': 'softmax', 'init_val': 1., 'use_l1_loss': False, 'use_div_loss': False,
                           'use_img_norm': args.method != 'labo_raw', 'use_txt_norm': args.method != 'labo_raw', 'view': 'global concept bottleneck',
                           'official_changes': ['ViT-B/32 instead of ViT-L/14', 'normalized CLIP features' if args.method != 'labo_raw' else 'official unnormalized embeddings retained',
                                                'MAFT stratified splits', 'aircraft class count 100 instead of config typo 102']})
        else:
            model = TipAdapterF(train, y, F.normalize(meta['class_text'].float(), dim=-1).cuda(), alpha=1., beta=1.)
            epochs, batch_size, lr = (20 if args.method == 'tip_f' else 0), 256, .001
            optimizer = torch.optim.AdamW([model.keys], lr=lr, eps=1e-4, weight_decay=.01)
            config.update({'optimizer': 'AdamW', 'eps': 1e-4, 'weight_decay': .01, 'init_alpha': 1., 'init_beta': 1.,
                           'search_scale': [50, 50], 'search_steps': [200, 20], 'cache_count': len(train),
                           'official_changes': ['all training examples instead of 16-shot', 'single cached center-crop cache',
                                                'validation checkpoint selection instead of test selection',
                                                'unified alpha/beta/search grid across datasets']})
        model.eval()
        config.update({'epochs': epochs, 'batch_size': batch_size, 'lr': lr})
        config['scheduler'] = 'constant' if args.method.startswith('labo') else 'cosine'
        history = []
        best = accuracy(model, val, vy)
        best_epoch = 0
        torch.save(snapshot(model), output / 'best.pt')
        history.append({'epoch': 0, 'val_accuracy': best})
        started = time.monotonic()
        steps = max(1, math.ceil(len(train) / batch_size))
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * steps))
                     if args.method == 'tip_f' else None)
        for epoch in range(1, epochs + 1):
            if args.method == 'coop':
                rate = 1e-5 if epoch == 1 else lr * .5 * (1 + math.cos(math.pi * (epoch - 2) / (epochs - 1)))
                for group in optimizer.param_groups:
                    group['lr'] = rate
            permutation = torch.randperm(len(train), device=train.device)
            losses = []
            for ids in permutation.split(batch_size):
                if args.method == 'coop':
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(model(train[ids]), y[ids])
                else:
                    loss = F.cross_entropy(model(train[ids]), y[ids])
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite training loss: {args.method} epoch {epoch}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                losses.append(loss.item())
            score = accuracy(model, val, vy)
            row = {'epoch': epoch, 'val_accuracy': score, 'train_loss': sum(losses) / len(losses),
                   'lr': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.monotonic() - started}
            history.append(row)
            if epoch % (50 if args.method.startswith('labo') else 5) == 0:
                print(json.dumps(row), flush=True)
            if score > best:
                best, best_epoch = score, epoch
                torch.save(snapshot(model), output / 'best.pt')
        state = torch.load(output / 'best.pt', weights_only=True)
        if args.method == 'coop':
            model.ctx.data.copy_(state['ctx'].cuda())
        else:
            model.load_state_dict(state)
        result = {'val_accuracy': best, 'best_epoch': best_epoch, 'history': history}
        if args.method.startswith('tip'):
            result['search'] = tip_search(model, val, vy)
            result['val_accuracy'] = result['search']['val_accuracy']
    result['config'] = config
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
