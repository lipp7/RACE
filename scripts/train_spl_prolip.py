"""All-shot cached-protocol ports; validation search, then separate full test."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from maft.recent_baselines import ProLIPProjection, proker_fit, proker_logits, rbf_kernel
from scripts.train import stratified_split

SOURCES = ('maft/recent_baselines.py', 'scripts/train_spl_prolip.py', 'scripts/extract_spl_features.py')
LRS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
LAMBDAS = (10., 1., .1, .01, .001, .0001, 0.)
BETAS = (.1, .5, 1., 2., 3., 5., 10.)
KERNEL_LAMBDAS = (.001, .01, .1, 1., 10.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_commit(name):
    return json.loads(Path("third_party/versions.json").read_text())[name]["commit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='prolip', choices=('prolip',))
    parser.add_argument('--dataset', required=True, choices=('cub', 'pets', 'flowers', 'cars', 'aircraft'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    assert args.method == 'prolip'
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(f'features/spl_fresh32_{args.dataset}')
    pre_root = Path(f'features/spl_preprojection10_{args.dataset}')
    data = torch.load((pre_root if args.method == 'prolip' else root)/'train.pt', weights_only=True)
    metadata = torch.load((pre_root if args.method == 'prolip' else root)/'metadata.pt', weights_only=False)
    text = F.normalize(metadata['class_text'].float(), dim=-1).cuda()
    ids, val_ids = stratified_split(data['label'], args.seed)
    feature = data['preprojection'].float() if args.method == 'prolip' else F.normalize(data['image'].float(), dim=-1)
    x, y = feature[ids].cuda(), data['label'][ids].cuda()
    vx, vy = feature[val_ids].cuda(), data['label'][val_ids].cuda()

    def predict(state, inputs):
        if args.method == 'prolip':
            return (F.normalize(inputs @ state['projection'].cuda(), dim=-1) @ text.T).argmax(-1)
        return torch.cat([proker_logits(z, x, text, state['coefficients'].cuda(), state['beta']).argmax(-1)
                          for z in inputs.split(512)])

    if args.evaluate:
        start = time.monotonic()
        result_hash = sha(output/'result.json')
        result = json.loads((output/'result.json').read_text())
        state = torch.load(output/'best.pt', weights_only=True)
        saved_split = torch.load(output/'split.pt', weights_only=True)
        assert torch.equal(ids, saved_split['train_ids']) and torch.equal(val_ids, saved_split['val_ids'])
        with torch.no_grad():
            val = predict(state, vx).cpu()
            val_acc = float((val == vy.cpu()).double().mean())
            assert abs(val_acc-result['val_accuracy']) < 1e-10, (val_acc, result['val_accuracy'])
            # First official test feature read in this script is after selection verification.
            test = torch.load((pre_root if args.method == 'prolip' else root)/'test.pt', weights_only=True)
            tx = test['preprojection'].float() if args.method == 'prolip' else F.normalize(test['image'].float(), dim=-1)
            pred = predict(state, tx.cuda()).cpu()
        assert sha(output/'result.json') == result_hash
        record = dict(accuracy=float((pred==test['label']).double().mean()), n=len(pred),
                      predictions=pred.tolist(), labels=test['label'].tolist(),
                      checkpoint_validation_accuracy=val_acc, selection_result_sha256=result_hash,
                      checkpoint_sha256=sha(output/'best.pt'), wall_seconds_including_load=time.monotonic()-start)
        (output/'evaluation_full.json').write_text(json.dumps(record, indent=2))
        print('FULL TEST', record['accuracy'], flush=True)
        return

    assert not (output/'result.json').exists(), 'Refusing to overwrite a completed selection'
    torch.save({'train_ids': ids, 'val_ids': val_ids}, output/'split.pt')
    config = dict(vars(args), protocol='all-shot B32 real ten-view mean preprojection port; unchanged grid and solver; not original few-shot numbers',
                  source_sha256={name:sha(name) for name in SOURCES},
                  reference_commit=reference_commit('ProLIP' if args.method=='prolip' else 'ProKeR'),
                  train_count=len(ids), val_count=len(val_ids), test_used_in_selection=False)
    trials, best = [], -1.
    torch.cuda.synchronize()
    started = time.monotonic()
    if args.method == 'prolip':
        config.update(epochs=300, batch_size='all training samples', optimizer='Adam eps=1e-4, no weight decay',
                      scheduler='cosine, 300 updates', learning_rates=LRS, regularizer_weights=LAMBDAS,
                      checkpoint_selection='final epoch for each candidate, validation chooses lr/lambda; author budget')
        initial = metadata['projection'].float().cuda()
        with torch.no_grad():
            initial_acc = float(((F.normalize(vx @ initial, dim=-1) @ text.T).argmax(-1)==vy).double().mean())
        config['initial_validation_accuracy'] = initial_acc
        for lr in LRS:
            for lmbda in LAMBDAS:
                model = ProLIPProjection(initial).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
                torch.cuda.synchronize()
                began = time.monotonic()
                first_loss, last_loss = None, None
                for epoch in range(300):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x, text), y) + lmbda*model.regularizer()
                    assert torch.isfinite(loss), (lr,lmbda,epoch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if epoch == 0: first_loss = float(loss.detach())
                last_loss = float(loss.detach())
                torch.cuda.synchronize()
                seconds = time.monotonic()-began
                with torch.no_grad():
                    acc = float((model(vx,text).argmax(-1)==vy).double().mean())
                row = dict(lr=lr, lmbda=lmbda, epochs=300, val_accuracy=acc, training_seconds=seconds,
                           first_loss=first_loss, last_loss=last_loss,
                           projection_displacement_norm=float((model.vit_proj-initial).norm().detach()))
                trials.append(row)
                if acc > best:
                    best, selected = acc, dict(lr=lr,lmbda=lmbda)
                    torch.save({'projection': model.vit_proj.detach().cpu()}, output/'best.pt')
                print(json.dumps(row), flush=True)
    else:
        config.update(beta_grid=BETAS, lambda_grid=KERNEL_LAMBDAS,
                      solver='FP32 author kernel, FP64 system assembly/Cholesky and residual prediction; no SGD epochs',
                      grid_note='uniform grid on all five datasets; author repository only includes RN50 grids and no CUB')
        with torch.no_grad():
            for beta in BETAS:
                for lmbda in KERNEL_LAMBDAS:
                    torch.cuda.synchronize(); began = time.monotonic()
                    coefficients = proker_fit(x, y, text, beta, lmbda)
                    logits = proker_logits(vx, x, text, coefficients, beta)
                    acc = float((logits.argmax(-1)==vy).double().mean())
                    residual = F.one_hot(y, len(text)).float()-x@text.T
                    matrix = rbf_kernel(x,x,beta).double()/lmbda
                    matrix.diagonal().add_(1.)
                    relative = float((matrix@coefficients-residual.double()).norm()/residual.double().norm())
                    assert relative < 1e-7, (beta,lmbda,relative)
                    torch.cuda.synchronize()
                    row = dict(beta=beta, lmbda=lmbda, val_accuracy=acc,
                               fitting_and_validation_seconds=time.monotonic()-began, relative_solve_residual=relative)
                    trials.append(row)
                    if acc > best:
                        best, selected = acc, dict(beta=beta,lmbda=lmbda)
                        torch.save({'coefficients':coefficients.cpu(),'beta':beta,'lmbda':lmbda}, output/'best.pt')
                    print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    result = dict(config=config, val_accuracy=best, selected=selected, history=trials,
                  search_wall_seconds_excluding_load=time.monotonic()-started,
                  split_sha256=sha(output/'split.pt'), checkpoint_sha256=sha(output/'best.pt'))
    (output/'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
