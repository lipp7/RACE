"""CoCoOp equation port: shared four context tokens plus instance Meta-Net bias."""
import torch
from torch import nn
from torch.nn import functional as F
from maft.reference_baselines import CoOp

class CoCoOp(CoOp):
    def __init__(self, clip_model, tokenizer, classes):
        super().__init__(clip_model,tokenizer,classes,n_ctx=4)
        tokens=tokenizer(['a photo of a']).to(self.ctx.device)
        with torch.no_grad():self.ctx.copy_(clip_model.token_embedding(tokens)[0,1:5])
        visual_dim=clip_model.visual.output_dim
        self.meta_net=nn.Sequential(nn.Linear(visual_dim,visual_dim//16),nn.ReLU(),nn.Linear(visual_dim//16,self.ctx.shape[-1]))
        self.meta_net.to(self.ctx.device)

    def conditioned_text_features(self, image_features, truncate=True):
        batch=len(image_features); classes=self.n_cls
        shifted=self.ctx[None]+self.meta_net(image_features)[:,None]
        prompts=torch.cat([self.prefix[None].expand(batch,-1,-1,-1),
                           shifted[:,None].expand(-1,classes,-1,-1),
                           self.suffix[None].expand(batch,-1,-1,-1)],dim=2)
        length=self.sequence_length if truncate else self.tokens.shape[1]
        x=prompts.reshape(batch*classes,-1,prompts.shape[-1])[:,:length]+self.clip.positional_embedding[:length]
        x=self.clip.transformer(x,attn_mask=self.clip.attn_mask[:length,:length])
        x=self.clip.ln_final(x)
        eot=self.tokens.argmax(-1)[None].expand(batch,-1).flatten()
        x=x[torch.arange(batch*classes,device=x.device),eot]@self.clip.text_projection
        return F.normalize(x.float(),dim=-1).reshape(batch,classes,-1)

    def forward(self,image_features):
        text=self.conditioned_text_features(image_features)
        return self.clip.logit_scale.exp()*torch.einsum('bd,bcd->bc',image_features,text)

    def prompt_state(self):
        return {'ctx':self.ctx.detach().cpu(),'meta_net':{k:v.detach().cpu() for k,v in self.meta_net.state_dict().items()}}

    def load_prompt_state(self,state):
        self.ctx.data.copy_(state['ctx'].to(self.ctx.device));self.meta_net.load_state_dict(state['meta_net'])
