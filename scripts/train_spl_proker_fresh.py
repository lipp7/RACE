"""E5 fresh crops, exact normalized ten-view mean, unchanged ProKeR grid."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from maft.recent_baselines import proker_fit,proker_logits,rbf_kernel
from scripts.train import stratified_split
from scripts.train_recent_baseline import BETAS,KERNEL_LAMBDAS,reference_commit
SOURCES=('scripts/train_spl_proker_fresh.py','maft/recent_baselines.py','scripts/train.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def feature(d):return F.normalize(F.normalize(d['views'].float(),dim=-1).mean(1),dim=-1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True);p.add_argument('--evaluate',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    cache=Path(f'features/spl_fresh32_{args.dataset}');data=torch.load(cache/'train.pt',weights_only=True);meta=torch.load(cache/'metadata.pt',weights_only=False)
    train,val=stratified_split(data['label'],args.seed);allx=feature(data);x=allx[train].cuda();y=data['label'][train].cuda();vx=allx[val].cuda();vy=data['label'][val].cuda();text=F.normalize(meta['class_text'].float(),dim=-1).cuda()
    config=dict(**vars(args),method='proker',views=10,aggregation='L2 normalize(mean of cached normalized global+9 local views)',
        beta_grid=BETAS,lambda_grid=KERNEL_LAMBDAS,reference_commit=reference_commit('ProKeR'),feature_metadata_sha256=sha(cache/'metadata.pt'),
        source_sha256={f:sha(f) for f in SOURCES},solver='Identical FP32 kernel, FP64 system assembly/Cholesky/prediction',selection='35 pairs validation only, same grid as existing global ProKeR')
    def predict(z,state):return torch.cat([proker_logits(c,x,text,state['coefficients'].cuda(),state['beta']).argmax(-1) for c in z.split(512)])
    with torch.no_grad():
        if args.evaluate:
            r=json.loads((root/'result.json').read_text());assert r['config']['source_sha256']==config['source_sha256'];state=torch.load(root/'best.pt',weights_only=True)
            accuracy=float((predict(vx,state)==vy).double().mean());assert abs(accuracy-r['val_accuracy'])<1e-10
            test=torch.load(cache/'test.pt',weights_only=True);pred=predict(feature(test).cuda(),state).cpu()
            record=dict(accuracy=float((pred==test['label']).double().mean()),n=len(pred),predictions=pred.tolist(),labels=test['label'].tolist(),
                checkpoint_validation_accuracy=accuracy,selection_result_sha256=sha(root/'result.json'),checkpoint_sha256=sha(root/'best.pt'))
            (root/'evaluation_full.json').write_text(json.dumps(record,indent=2));print('TEST',record['accuracy'],flush=True);return
        assert not (root/'result.json').exists()
        torch.save(dict(train_ids=train,val_ids=val),root/'split.pt');history=[];best=-1;start=time.monotonic()
        for beta in BETAS:
            for lmbda in KERNEL_LAMBDAS:
                coefficients=proker_fit(x,y,text,beta,lmbda);logits=proker_logits(vx,x,text,coefficients,beta)
                acc=float((logits.argmax(-1)==vy).double().mean());residual=F.one_hot(y,len(text)).float()-x@text.T
                matrix=rbf_kernel(x,x,beta).double()/lmbda;matrix.diagonal().add_(1.)
                relative=float((matrix@coefficients-residual.double()).norm()/residual.double().norm());assert relative<1e-7,(beta,lmbda,relative)
                row=dict(beta=beta,lmbda=lmbda,val_accuracy=acc,relative_solve_residual=relative);history.append(row)
                if acc>best:
                    best=acc;selected=dict(beta=beta,lmbda=lmbda);torch.save(dict(coefficients=coefficients.cpu(),beta=beta,lmbda=lmbda),root/'best.pt')
                print(json.dumps(row),flush=True)
        (root/'result.json').write_text(json.dumps(dict(config=config,val_accuracy=best,selected=selected,history=history,seconds=time.monotonic()-start,
            train_count=len(train),val_count=len(val),split_sha256=sha(root/'split.pt'),checkpoint_sha256=sha(root/'best.pt')),indent=2))
if __name__=='__main__':main()
