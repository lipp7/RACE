"""Evaluate validation-frozen reference baselines on the full official test set."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.train_reference import reference_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    root = Path(args.run)
    result_path = root / 'result.json'
    before = hashlib.sha256(result_path.read_bytes()).hexdigest()
    result = json.loads(result_path.read_text())
    cfg = result['config']
    state = torch.load(root / 'best.pt', weights_only=True)
    state = {k: v.cuda() for k, v in state.items()}
    method = cfg['method']
    if method.startswith('labo'):
        text = torch.load(cfg['concept_path'], weights_only=True)['text'].cuda()
    def predict(features):
        predictions = []
        with torch.no_grad():
            for batch in features.split(256):
                if method == 'coop':
                    logits = batch @ state['text_features'].T
                elif method == 'linear':
                    logits = F.linear(batch, state['weight'], state['bias'])
                elif method.startswith('labo'):
                    logits = 100 * ((batch @ text.T) @ state['asso_mat'].softmax(-1).T)
                elif method.startswith('tip'):
                    alpha, beta = result['search']['alpha'], result['search']['beta']
                    logits = 100 * batch @ state['text_features'].T
                    cache_logits = (beta * (batch @ state['keys'].T - 1)).exp() @ state['values']
                    logits += alpha * cache_logits
                else:
                    raise ValueError(method)
                assert torch.isfinite(logits).all()
                predictions.append(logits.argmax(-1).cpu())
        return torch.cat(predictions)
    feature_root = (f"features/labo_raw_{cfg['dataset']}/raw_" if method == 'labo_raw'
                    else str(Path(cfg.get("features") or f"features/{cfg['dataset']}_views_grid3")) + "/")
    development = torch.load(feature_root + 'train.pt', weights_only=True)
    split = torch.load(root / 'split.pt', weights_only=True)
    val_ids = split['val_ids']
    val_features = (development['image'].float() if method == 'labo_raw'
                    else reference_features(development, cfg.get('feature_mode', 'global')))[val_ids]
    val_features = val_features.cuda()
    val_predictions = predict(val_features)
    val_accuracy = (val_predictions == development['label'][val_ids]).float().mean().item()
    assert abs(val_accuracy - result['val_accuracy']) < 1e-6, (val_accuracy, result['val_accuracy'])
    data = torch.load(feature_root + 'test.pt', weights_only=True)
    features = (data['image'].float() if method == 'labo_raw'
                else reference_features(data, cfg.get('feature_mode', 'global')))
    features = features.cuda()
    labels = data['label']
    predictions = predict(features)
    score = (predictions == labels).float().mean().item()
    class_scores = [(predictions[labels == c] == c).float().mean().item() for c in labels.unique()]
    record = {'accuracy': score, 'mean_class_accuracy': sum(class_scores) / len(class_scores),
              'n': len(labels), 'predictions': predictions.tolist(), 'labels': labels.tolist(),
              'selection_result_sha256': before, 'method': method, 'dataset': cfg['dataset'],
              'checkpoint_validation_accuracy': val_accuracy}
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == before
    (root / 'evaluation_full.json').write_text(json.dumps(record))
    print(json.dumps({k: v for k, v in record.items() if k not in ('predictions', 'labels')}), flush=True)


if __name__ == '__main__':
    main()
