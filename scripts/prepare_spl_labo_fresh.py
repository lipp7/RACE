"""Recover raw ten-view means for LaBo and verify the exact E5 crop encoding."""
import json,hashlib,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scripts.extract_spl_features import dataset
from scripts.prepare_labo import greedy_selection,PER_CLASS,FACILITY
from scripts.train import stratified_split
from scripts.run_spl_experiments import freeze

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import open_clip
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 encoder,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32-quickgelu',pretrained='openai', cache_dir='cache/open_clip');encoder=encoder.cuda().eval();projection=encoder.visual.proj.detach().clone();encoder.visual.proj=None
 for d in ('cub','flowers','aircraft'):
  root=Path(f'features/spl_labo_fresh10_{d}');root.mkdir(exist_ok=True);fresh=Path(f'features/spl_fresh32_{d}');candidate=Path(f'features/labo_raw_{d}/candidates.pt')
  freeze(root/'protocol.json',dict(dataset=d,views=10,image_normalization=False,text_normalization=False,aggregation='Arithmetic mean of ten unnormalized FP32 projected CLIP features; compare per-view normalized encoding to same E5 fresh cache',precision='FP32 image, existing official candidate text bfloat16-encoded float32 cache',candidate_source=str(candidate),candidate_sha256=sha(candidate),selection='Per-seed training images only; MI + facility, same official candidates and per-class settings as historical LaBo',source_sha256={s:sha(s) for s in ('scripts/prepare_spl_labo_fresh.py','scripts/prepare_labo.py','scripts/extract_spl_features.py','scripts/train.py')}))
  meta=torch.load(fresh/'metadata.pt',weights_only=False);torch.save(meta,root/'metadata.pt');audits=[]
  for split in ('train','test'):
   dest=root/f'{split}.pt'
   if dest.exists():continue
   original=torch.load(fresh/f'{split}.pt',weights_only=True);loader=DataLoader(dataset(d,split,preprocess),batch_size=16,num_workers=4,shuffle=False,pin_memory=True)
   values=[];labels=[];offset=0;max_error=0.;min_cosine=1.;equal=0;coordinates=0;start=time.monotonic()
   with torch.no_grad():
    for views,y in loader:
     x=views.flatten(0,1).cuda(non_blocking=True);n=len(y)
     h=torch.cat([encoder.encode_image(t).float() for t in x.split(32)]).reshape(n,10,-1);projected=h@projection
     normalized=F.normalize(projected,dim=-1).cpu().half();target=original['views'][offset:offset+n]
     error=float((normalized.float()-target.float()).abs().max());max_error=max(max_error,error)
     cosine=float(F.cosine_similarity(normalized.float(),target.float(),dim=-1).min());min_cosine=min(min_cosine,cosine)
     equal+=int((normalized==target).sum());coordinates+=normalized.numel()
     assert error<=1e-3 and cosine>1-1e-5,(d,split,offset,error,cosine)
     assert torch.equal(y,original['label'][offset:offset+n])
     values.append(projected.mean(1).cpu());labels.append(y);offset+=n
     if offset%480==0:print('RAW FRESH',d,split,offset,'/',len(original['label']),flush=True)
   assert offset==len(original['label']);torch.save(dict(image=torch.cat(values),label=torch.cat(labels)),dest)
   audit=dict(split=split,n=offset,max_coordinate_error=max_error,min_cosine=min_cosine,bitwise_equal_fraction=equal/coordinates,seconds=time.monotonic()-start,fresh_reference_sha256=sha(fresh/f'{split}.pt'));audits.append(audit)
   print('RAW AUDIT',d,json.dumps(audit),flush=True)
  if audits:(root/'extraction.json').write_text(json.dumps(dict(records=audits),indent=2))
  candidates=torch.load(candidate,weights_only=True);assert candidates['classes']==meta['classes'];source=torch.load(root/'train.pt',weights_only=True)
  for seed in (1,2,3):
   selected=root/f'selected_s{seed}.pt'
   if selected.exists():continue
   tr,va=stratified_split(source['label'],seed)
   ids,mi=greedy_selection(source['image'][tr].cuda(),source['label'][tr].cuda(),candidates['text'].cuda(),candidates['owners'].cuda(),PER_CLASS[d],FACILITY[d])
   torch.save(dict(text=candidates['text'][ids],owners=candidates['owners'][ids],indices=ids,concepts=[candidates['concepts'][i] for i in ids.tolist()],train_ids=tr,val_ids=va,mi=mi[ids]),selected)
   print('FRESH CONCEPTS',d,seed,len(ids),flush=True)

if __name__=='__main__':main()
