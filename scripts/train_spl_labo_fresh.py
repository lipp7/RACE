"""LaBo with official raw convention and training-only fresh ten-view selection."""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from maft.reference_baselines import LaBo
from scripts.train import stratified_split

SOURCES=('scripts/train_spl_labo_fresh.py','scripts/prepare_spl_labo_fresh.py','scripts/prepare_labo.py','maft/reference_baselines.py','scripts/train.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def predict(model,x):return torch.cat([model(v).argmax(-1).cpu() for v in x.split(256)])

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('cub','flowers','aircraft'));p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True);p.add_argument('--evaluate',action='store_true');args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
 root=Path(f'features/spl_labo_fresh10_{args.dataset}');output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 data=torch.load(root/'train.pt',weights_only=True);meta=torch.load(root/'metadata.pt',weights_only=False);tr,va=stratified_split(data['label'],args.seed)
 cp=root/f'selected_s{args.seed}.pt';concepts=torch.load(cp,weights_only=True);assert torch.equal(tr,concepts['train_ids']) and torch.equal(va,concepts['val_ids'])
 text=concepts['text'].cuda();model=LaBo(concepts['owners'].cuda(),len(meta['classes']));x=data['image'].cuda()@text.T;labels=data['label'].cuda()
 if args.evaluate:
  result=json.loads((output/'result.json').read_text());result_hash=sha(output/'result.json')
  assert result['config']['source_sha256']=={s:sha(s) for s in SOURCES};assert result['config']['concept_sha256']==sha(cp)
  model.load_state_dict(torch.load(output/'best.pt',weights_only=True));model.eval();pred=predict(model,x[va]);v=float((pred==data['label'][va]).double().mean());assert abs(v-result['val_accuracy'])<1e-7
  test=torch.load(root/'test.pt',weights_only=True);features=test['image'].cuda()@text.T;pred=predict(model,features);y=test['label']
  record=dict(accuracy=float((pred==y).double().mean()),n=len(y),predictions=pred.tolist(),labels=y.tolist(),checkpoint_validation_accuracy=v,selection_result_sha256=result_hash,checkpoint_sha256=sha(output/'best.pt'),split_sha256=sha(output/'split.pt'),concept_sha256=sha(cp),features=str(root),view_count=10)
  assert result_hash==sha(output/'result.json');(output/'evaluation_full.json').write_text(json.dumps(record,indent=2));print('FULL TEST',record['accuracy'],flush=True);return
 assert not (output/'result.json').exists(),'Completed selection is immutable'
 epochs=5000 if args.dataset=='cub' else 10000;batch=512 if args.dataset=='cub' else 256;lr=1e-5 if args.dataset=='flowers' else 5e-5
 optimizer=torch.optim.Adam(model.parameters(),lr=lr);torch.save(dict(train_ids=tr,val_ids=va),output/'split.pt')
 config=dict(method='labo',dataset=args.dataset,seed=args.seed,features=str(root),views=10,aggregation='mean of unnormalized projected image features',use_img_norm=False,use_txt_norm=False,concept_path=str(cp),concept_sha256=sha(cp),concept_count=len(text),optimizer='Adam',lr=lr,epochs=epochs,batch_size=batch,scheduler='constant',asso_act='softmax',init_val=1.,source_sha256={s:sha(s) for s in SOURCES},train_count=len(tr),val_count=len(va),selection='validation only, earliest checkpoint wins ties including epoch0',concept_selection='training-only fresh features; same official candidates, MI/facility and per-class budgets',test_used_in_selection=False)
 train=x[tr];y=labels[tr];val=x[va];best=float((predict(model,val)==data['label'][va]).double().mean());best_epoch=0;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/'best.pt');history=[dict(epoch=0,val_accuracy=best)];start=time.monotonic()
 for epoch in range(1,epochs+1):
  losses=[]
  for ids in torch.randperm(len(train),device='cuda').split(batch):
   optimizer.zero_grad();loss=F.cross_entropy(model(train[ids]),y[ids]);loss.backward();optimizer.step();losses.append(loss.detach())
  v=float((predict(model,val)==data['label'][va]).double().mean());value=float(torch.stack(losses).mean());assert math.isfinite(value)
  row=dict(epoch=epoch,val_accuracy=v,train_loss=value,lr=lr,elapsed_seconds=time.monotonic()-start);history.append(row)
  if v>best:best=v;best_epoch=epoch;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/'best.pt')
  if epoch%100==0:print(json.dumps(row),flush=True)
 result=dict(config=config,val_accuracy=best,best_epoch=best_epoch,history=history,seconds=time.monotonic()-start)
 (output/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
