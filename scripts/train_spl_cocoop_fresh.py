"""CoCoOp on E5 fresh ten-view means; unchanged batch-one ten-epoch solver."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import open_clip
import torch
from torch.nn import functional as F
from maft.cocoop import CoCoOp
from scripts.train import stratified_split
from scripts.train_reference import reference_features
from scripts.train_cocoop import predictions

SOURCES=('scripts/train_spl_cocoop_fresh.py','maft/cocoop.py','maft/reference_baselines.py','scripts/train_cocoop.py','scripts/train_reference.py','scripts/train.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True);p.add_argument('--evaluate',action='store_true');args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
 root=Path(f'features/spl_fresh32_{args.dataset}');output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 meta=torch.load(root/'metadata.pt',weights_only=False);data=torch.load(root/'train.pt',weights_only=True)
 features=reference_features(data,'viewmean').cuda();train,val=stratified_split(data['label'],args.seed);labels=data['label'].cuda()
 clip=open_clip.create_model(meta['model'],pretrained=meta['pretrained'], cache_dir='cache/open_clip').eval().cuda();model=CoCoOp(clip,open_clip.get_tokenizer(meta['model']),meta['classes'])
 if args.evaluate:
  result=json.loads((output/'result.json').read_text());result_hash=sha(output/'result.json')
  assert result['config']['source_sha256']=={s:sha(s) for s in SOURCES}
  split=torch.load(output/'split.pt',weights_only=True);assert torch.equal(train,split['train_ids']) and torch.equal(val,split['val_ids'])
  model.load_prompt_state(torch.load(output/'best.pt',weights_only=True));model.eval()
  pred=predictions(model,features[val]);va=float((pred==data['label'][val]).double().mean());assert abs(va-result['val_accuracy'])<1e-6
  test=torch.load(root/'test.pt',weights_only=True);pred=predictions(model,reference_features(test,'viewmean').cuda());y=test['label']
  evaluation=dict(accuracy=float((pred==y).double().mean()),n=len(y),predictions=pred.tolist(),labels=y.tolist(),checkpoint_validation_accuracy=va,selection_result_sha256=result_hash,checkpoint_sha256=sha(output/'best.pt'),split_sha256=sha(output/'split.pt'),features=str(root),view_count=10)
  assert sha(output/'result.json')==result_hash
  (output/'evaluation_full.json').write_text(json.dumps(evaluation,indent=2));print('FULL TEST',evaluation['accuracy'],flush=True);return
 assert not (output/'result.json').exists(),'Completed selection is immutable'
 parameters=[v for v in model.parameters() if v.requires_grad];optimizer=torch.optim.SGD(parameters,lr=.002,momentum=.9,weight_decay=.0005)
 config=dict(method='cocoop',dataset=args.dataset,seed=args.seed,features=str(root),feature_mode='viewmean',views=10,epochs=10,batch_size=1,context_tokens=4,context_init='a photo of a',meta_net_hidden=32,optimizer='SGD',lr=.002,momentum=.9,weight_decay=.0005,precision='bf16 autocast',backbone=meta['model'],pretrained=meta['pretrained'],train_count=len(train),val_count=len(val),source_sha256={s:sha(s) for s in SOURCES},selection='validation only, earliest checkpoint wins ties; initial epoch included',online_augmentation=False,changes='All-shot B32 cached normalized ten-view mean port, same solver as historical CoCoOp; not original base-to-new protocol')
 torch.save(dict(train_ids=train,val_ids=val),output/'split.pt')
 best=float((predictions(model,features[val])==data['label'][val]).double().mean());best_epoch=0;torch.save(model.prompt_state(),output/'best.pt');history=[dict(epoch=0,val_accuracy=best)];started=time.monotonic()
 for epoch in range(1,11):
  rate=1e-5 if epoch==1 else .002*.5*(1+math.cos(math.pi*(epoch-2)/9))
  for group in optimizer.param_groups:group['lr']=rate
  loss_total=0.
  for idx in train[torch.randperm(len(train))]:
   optimizer.zero_grad()
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(features[idx:idx+1]),labels[idx:idx+1])
   loss.backward();optimizer.step();loss_total+=loss.detach()
  va=float((predictions(model,features[val])==data['label'][val]).double().mean())
  row=dict(epoch=epoch,val_accuracy=va,train_loss=float(loss_total/len(train)),elapsed_seconds=time.monotonic()-started);history.append(row);print(json.dumps(row),flush=True)
  assert math.isfinite(row['train_loss'])
  if va>best:best=va;best_epoch=epoch;torch.save(model.prompt_state(),output/'best.pt')
 (output/'result.json').write_text(json.dumps(dict(config=config,val_accuracy=best,best_epoch=best_epoch,history=history,seconds=time.monotonic()-started,trainable_parameters=sum(v.numel() for v in parameters)),indent=2))

if __name__=='__main__':main()
