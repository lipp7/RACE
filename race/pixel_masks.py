"""Exact pixel mask construction used by the CUB intervention analysis."""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
POLICIES = ('private', 'shared', 'random_crop', 'background')
FRACTIONS = (.1, .2, .3)

def project(weights,rectangles,w,h):
    heat=np.zeros((h,w),dtype=np.float32)
    for value,(l,t,r,b) in zip(weights,rectangles):
        l,t=max(0,int(np.ceil(l))),max(0,int(np.ceil(t)));r,b=min(w,int(np.ceil(r))),min(h,int(np.ceil(b)))
        if r>l and b>t:heat[t:b,l:r]+=value/((r-l)*(b-t))
    return heat

def views(image,preprocess):
    w,h=image.size;cw,ch=w//2,h//2;items=[preprocess(image)]
    for row in range(3):
        top=round(row*(h-ch)/2)
        for col in range(3):
            left=round(col*(w-cw)/2);items.append(preprocess(image.crop((left,top,left+cw,top+ch))))
    return torch.stack(items)

class PixelDataset(Dataset):
    def __init__(self,ids,paths,rectangles,weights,boxes,preprocess,original=False):
        self.ids,self.paths,self.rectangles,self.weights,self.boxes,self.preprocess=ids,paths,rectangles,weights,boxes,preprocess
        self.original=original
        self.rows=[(i,p,f) for i in ids for p in POLICIES for f in FRACTIONS] if not original else [(i,'original',0.) for i in ids]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        i,policy,f=self.rows[index]
        with Image.open(self.paths[i]) as opened:image=opened.convert('RGB')
        w,h=image.size
        if self.original:return views(image,self.preprocess).unsqueeze(0),i,policy,f
        rng=np.random.default_rng(20261015+i);tie=rng.random((h,w))
        if policy in ('private','shared'):heat=project(self.weights[policy][i],self.rectangles[i],w,h)
        elif policy=='random_crop':heat=project(rng.dirichlet(np.ones(9)),self.rectangles[i],w,h)
        else:
            x,y,bw,bh=self.boxes[i];xx,yy=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
            heat=((xx<x)|(xx>x+bw)|(yy<y)|(yy>y+bh)).astype(np.float32)
        # Lexicographic ties retain the actual score ordering; identical random tie map per image.
        order=np.lexsort((tie.ravel(),heat.ravel()));n=round(f*w*h)
        mask=np.zeros(w*h,dtype=bool);mask[order[-n:]]=True;mask=mask.reshape(h,w)
        rgb=np.asarray(image).copy();fill=np.rint(rgb.reshape(-1,3).mean(0)).astype(np.uint8)
        delete=rgb.copy();delete[mask]=fill;retain=np.broadcast_to(fill,rgb.shape).copy();retain[mask]=rgb[mask]
        return torch.stack([views(Image.fromarray(delete),self.preprocess),views(Image.fromarray(retain),self.preprocess)]),i,policy,f
