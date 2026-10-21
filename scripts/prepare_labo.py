"""Encode official LaBo candidates; select concepts separately on training splits.

Use the exact MI formula and greedy MI+facility-location objective from the
official source, with cosine distance on the same MI-augmented text features.
This implementation removes the Apricot dependency and records the objective.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scripts.train import stratified_split
from maft.protocol import fewshot_ids

OFFICIAL = {'cub': 'CUB', 'flowers': 'flower', 'aircraft': 'aircraft'}
FACILITY = {'cub': .1, 'flowers': 1., 'aircraft': .5}
PER_CLASS = {'cub': 50, 'flowers': 25, 'aircraft': 50}


class GlobalImages(Dataset):
    def __init__(self, dataset, split, preprocess):
        self.dataset, self.preprocess = dataset, preprocess
        if dataset == 'cub':
            from maft.data import CUBImages
            self.items = CUBImages('data/CUB_200_2011', split, preprocess).items
        elif dataset == 'flowers':
            from scripts.extract_flowers import FlowerViews
            splits = ('train', 'val') if split == 'train' else ('test',)
            self.items = FlowerViews('data', splits, preprocess).items
        else:
            from scripts.extract_transport import TransportViews
            source_split = 'trainval' if split == 'train' else 'test'
            self.items = TransportViews('data', 'aircraft', source_split, preprocess).items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        if self.dataset == 'aircraft':
            image = image.crop((0, 0, image.width, image.height - 20))
        return self.preprocess(image), label


def greedy_selection(features, labels, text, owners, count, facility_weight):
    n_classes = int(labels.max()) + 1
    means = torch.stack([features[labels == c].mean(0) for c in range(n_classes)])
    scores = text @ means.T
    normalized = scores / (scores.sum(dim=0) * n_classes)
    marginal = normalized.sum(dim=1, keepdim=True)
    mi = (normalized * torch.log(normalized / (marginal / n_classes))).sum(dim=1)
    if not torch.isfinite(mi).all():
        raise ValueError('Official MI formula produced nonfinite scores; inspect features before selecting.')
    selected = []
    for c in range(n_classes):
        ids = (owners == c).nonzero(as_tuple=True)[0]
        if len(ids) <= count:
            selected.extend(ids.tolist())
            continue
        modular = mi[ids] * 1e7
        augmented = torch.cat([modular[:, None], text[ids]], dim=1)
        cosine = F.normalize(augmented, dim=-1) @ F.normalize(augmented, dim=-1).T
        distances = (1 - cosine).clamp_min(0)
        similarities = distances.max() - distances
        coverage = torch.zeros(len(ids), device=text.device)
        available = torch.ones(len(ids), dtype=torch.bool, device=text.device)
        for _ in range(count):
            gains = modular + facility_weight * (torch.maximum(coverage[None], similarities) - coverage).sum(1)
            gains = gains.masked_fill(~available, float('-inf'))
            winner = int(gains.argmax())
            available[winner] = False
            coverage = torch.maximum(coverage, similarities[winner])
            selected.append(int(ids[winner]))
    return torch.tensor(selected), mi.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=OFFICIAL, required=True)
    p.add_argument('--seeds', default='1,2,3')
    p.add_argument('--shots',type=int,default=0)
    p.add_argument('--raw', action='store_true', help='Official default: unnormalized image/text features')
    args = p.parse_args()
    source = Path('third_party/LaBo')
    concepts_file = source / f'datasets/{OFFICIAL[args.dataset]}/concepts/class2concepts.json'
    features = Path(f'features/spl_fresh32_{args.dataset}')
    output = Path(f"features/labo_{'raw_' if args.raw else ''}{args.dataset}")
    output.mkdir(parents=True, exist_ok=True)
    meta = torch.load(features / 'metadata.pt', weights_only=False)
    classes = meta['classes']
    def canonical(name):
        return name.lower().replace('_', ' ').rstrip('?')
    mapping = {canonical(c): i for i, c in enumerate(classes)}
    original = json.loads(concepts_file.read_text())
    official_names = {canonical(name) for name in original}
    assert set(mapping) == official_names, (set(mapping) - official_names, official_names - set(mapping))
    raw, owner = [], []
    for name in sorted(original):
        raw.extend(original[name])
        owner.extend([mapping[canonical(name)]] * len(original[name]))
    texts, unique_ids = np.unique(np.array(raw), return_index=True)
    owners = torch.tensor(owner)[torch.from_numpy(unique_ids)]
    cache = output / 'candidates.pt'
    device = 'cuda'
    need_images = args.raw and any(not (output / f'raw_{s}.pt').exists() for s in ('train', 'test'))
    if not cache.exists() or need_images:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(meta['model'], pretrained=meta['pretrained'], cache_dir='cache/open_clip')
        model = model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(meta['model'])
    if not cache.exists():
        chunks = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(texts), 256):
                batch = tokenizer(texts[start:start + 256].tolist()).to(device)
                encoded = model.encode_text(batch).float()
                chunks.append((encoded if args.raw else F.normalize(encoded, dim=-1)).cpu())
                if start % 4096 == 0:
                    print(args.dataset, 'text encoding', start, '/', len(texts), flush=True)
        torch.save({'text': torch.cat(chunks), 'owners': owners, 'concepts': texts.tolist(), 'classes': classes}, cache)
    if need_images:
        for split in ('train', 'test'):
            path = output / f'raw_{split}.pt'
            if path.exists():
                continue
            loader = DataLoader(GlobalImages(args.dataset, split, preprocess), batch_size=128,
                                num_workers=8, shuffle=False, pin_memory=True)
            image_chunks, label_chunks = [], []
            # Full float32 image encoding preserves the original model feature scale.
            with torch.no_grad():
                for index, (image, label) in enumerate(loader):
                    image_chunks.append(model.encode_image(image.cuda()).float().cpu())
                    label_chunks.append(label)
            raw_data = {'image': torch.cat(image_chunks), 'label': torch.cat(label_chunks)}
            original_data = torch.load(features / f'{split}.pt', weights_only=True)
            assert torch.equal(raw_data['label'], original_data['label'])
            # Confirm identical image ordering and global preprocessing, allowing the
            # half-precision storage used by the existing cached feature files.
            error = (F.normalize(raw_data['image'], dim=-1) - F.normalize(original_data['image'].float(), dim=-1)).abs().max().item()
            assert error < .002, (args.dataset, split, error)
            torch.save(raw_data, path)
            print('Raw image features', args.dataset, split, len(raw_data['label']), 'direction error', error, flush=True)
    candidates = torch.load(cache, weights_only=True)
    assert candidates['concepts'] == texts.tolist()
    data = torch.load(features / 'train.pt', weights_only=True)
    # Use global features from the same cached representation used by MAFT.
    images = (torch.load(output / 'raw_train.pt', weights_only=True)['image'] if args.raw
              else F.normalize(data['image'].float(), dim=-1))
    labels = data['label']
    for seed in map(int, args.seeds.split(',')):
        tag=f'fs{args.shots}_' if args.shots else ''
        path = output / f'selected_{tag}s{seed}.pt'
        if path.exists():
            continue
        train, val = stratified_split(labels, seed)
        train = fewshot_ids(labels,train,args.shots,seed)
        ids, mi = greedy_selection(images[train].to(device), labels[train].to(device),
                                  candidates['text'].to(device), owners.to(device),
                                  PER_CLASS[args.dataset], FACILITY[args.dataset])
        torch.save({'text': candidates['text'][ids], 'owners': owners[ids], 'indices': ids,
                    'concepts': [str(texts[i]) for i in ids.tolist()], 'train_ids': train, 'val_ids': val,
                    'mi': mi[ids]}, path)
        print('Selected', args.dataset, seed, len(ids), flush=True)
    provenance = {
        'official_commit': json.loads(Path('third_party/versions.json').read_text())['LaBo']['commit'],
        'candidate_sha256': hashlib.sha256(concepts_file.read_bytes()).hexdigest(),
        'source': str(concepts_file), 'raw_count': len(raw), 'unique_count': len(texts),
        'concepts_per_class': PER_CLASS[args.dataset], 'submodular_weights': [1e7, FACILITY[args.dataset]],
        'backbone': meta['model'], 'pretrained': meta['pretrained'], 'use_img_norm': not args.raw, 'use_txt_norm': not args.raw,
        'selection_uses': 'training images only; no validation or test images',
        'official_protocol_changes': ['ViT-B/32 instead of ViT-L/14', 'normalized image/text embeddings' if not args.raw else 'official unnormalized embedding convention retained',
                                      'MAFT stratified development split', 'correct aircraft class count from 102 to 100'],
    }
    (output / ('provenance.json' if not args.shots else f'provenance_fs{args.shots}.json')).write_text(json.dumps({**provenance,'shots':args.shots}, indent=2))


if __name__ == '__main__':
    main()
