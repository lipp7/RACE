"""Isolated SPL supplement variants. Frozen historical implementations stay intact."""
import torch
from torch.nn import functional as F
from scripts.train_attack_defense_combination import make_model
from maft.attack_defense_views import attack_defense_loss

STRUCTURES=('full','static','no_conditioner','no_visibility','no_concept','uniform_local','class_static_concept')
OPPONENTS=('hardest','fixed_random','semantic_nn','topk2_mix','topk3_mix')
EVIDENCES=('full','shared_as_opponent','no_support','no_shared_residual','no_direction')

def build(meta, config):
    mode=config.get('mode','smooth');model=make_model(meta,mode,64)
    structure=config.get('structure','full');assert structure in STRUCTURES
    if structure=='static':model.prompt_mode='static'
    elif structure=='no_conditioner':model.use_conditioner=False
    elif structure=='no_visibility':model.visual_relevance_weight=0.
    elif structure=='no_concept':model.semantic_scale=0.
    elif structure=='uniform_local':model.local_mode='mean'
    model.evidence_temperature=float(config.get('evidence_temperature',.3))
    model.support_temperature=float(config.get('support_temperature',.5))
    opponent=config.get('opponent','hardest');assert opponent in OPPONENTS
    if opponent in ('fixed_random','semantic_nn'):
        model.rival_mode='fixed'
        if opponent=='semantic_nn':
            similarities=model.class_text@model.class_text.T
            similarities.fill_diagonal_(-torch.inf)
            model.fixed_rivals.copy_(similarities.argmax(-1))
    elif opponent.startswith('topk'):model.opponent_count=int(opponent[4])
    evidence=config.get('evidence','full');assert evidence in EVIDENCES
    if evidence=='shared_as_opponent':model.negative_mode='opponent'
    if evidence=='no_support':model.support_temperature=0.
    def modify(module,args,out):
        if structure=='class_static_concept':
            # Image-independent, class-only concept policy as specified in E2.
            m=torch.softmax(module.class_text@module.concept_text.T/module.concept_temperature+module.concept_delta,-1)
            m=m[None].expand(len(args[0]),-1,-1)
            response=out['local_views']@module.concept_text.T
            local_scores=module.head(out['local_views']).transpose(1,2)
            gate=local_scores+module.semantic_scale*torch.einsum('bck,brk->bcr',m,response)
            attention=torch.softmax(gate/module.view_temperature,-1)
            base=module.head(_adapted_global(module,args[0]))+module.beta*(attention*local_scores).sum(-1)
            out.update(concept_weights=m,gate=gate,attention=attention,base_logits=base)
            if mode!='ng':
                masked=base.detach()[:,None].expand(-1,len(module.class_text),-1).clone()
                masked.diagonal(dim1=1,dim2=2).fill_(-torch.inf)
                rival=masked.argmax(-1)
                if module.rival_mode=='fixed':rival=module.fixed_rivals[None].expand(len(base),-1)
                out['rival_ids']=rival
                plus,shared,support,utility=_evidence(module,out)
                out.update(plus=plus,minus=shared,support_mass=support,identity=utility)
            else:out['logits']=base
        if mode!='ng':
            support=out.get('support_mass',torch.ones_like(out['identity']))
            out['support_mass']=support
            if evidence in ('no_shared_residual','no_direction'):
                other=out.get('opponent_anchor')
                if other is None:other=out['anchor'].gather(1,out['rival_ids'][:,:,None].expand_as(out['anchor']))
                direction=out['anchor'] if evidence=='no_direction' else F.normalize(out['anchor']-other,dim=-1)
                residual=out['plus'] if evidence=='no_shared_residual' else out['plus']-out['minus']
                out['identity']=support*(residual*direction).sum(-1)
            out['logits']=out['base_logits']+out['effective_scale']*out['identity']
        return out
    if structure=='class_static_concept' or evidence!='full':model.register_forward_hook(modify)
    return model

def _adapted_global(model,views):
    # Avoid a second adapter call: reconstruct global feature from the hook in trainer
    # when available; standalone forward uses the same residual mapping.
    original=F.normalize(views.float(),dim=-1)[:,0]
    return F.normalize(original+model.visual_adapter(original),dim=-1)

def _evidence(model,out):
    gate=out['gate'];ids=out['rival_ids'];rival=gate.gather(1,ids[:,:,None].expand_as(gate))
    difference=gate-rival
    plus=F.normalize(torch.softmax(difference/model.evidence_temperature,-1)@out['local_views'],dim=-1)
    negative=torch.minimum(gate,rival) if model.negative_mode=='shared' else -difference
    shared=F.normalize(torch.softmax(negative/model.evidence_temperature,-1)@out['local_views'],dim=-1)
    support=torch.sigmoid(difference/model.support_temperature).mean(-1) if model.support_temperature>0 else torch.ones_like(difference[:,:,0])
    other=out['anchor'].gather(1,ids[:,:,None].expand_as(out['anchor']))
    utility=support*((plus-shared)*F.normalize(out['anchor']-other,dim=-1)).sum(-1)
    return plus,shared,support,utility

def forward_features(model,views):
    # The class-static hook uses an extra adapter call for the global feature.
    captured=[]
    h=model.visual_adapter.register_forward_hook(lambda m,a,d:captured.append((a[0],d)))
    try:out=model(views)
    finally:h.remove()
    original,delta=captured[0];assert original.ndim==3
    return out,original.detach(),F.normalize(original+delta,dim=-1)

def objective(out,labels,config):
    ce=F.cross_entropy(out['logits'],labels)
    if config.get('mode','smooth')=='ng':return ce,dict(ce=ce.detach(),identity=ce.detach()*0,attack=ce.detach()*0,neutral=ce.detach()*0)
    rows=torch.arange(len(labels),device=labels.device)
    j=out['rival_ids'][rows,labels]
    own=out['anchor'][rows,labels]
    other=out['opponent_anchor'][rows,labels] if out.get('opponent_count',1)>1 else out['anchor'][rows,j]
    shared=out['minus'][rows,labels]
    # Reuse the unchanged identity/attack implementation, including its exact
    # normalization and directed-primary-rival convention for top-k mixtures.
    base,parts=attack_defense_loss(out,labels,main_weight=config.get('identity_weight',.3),
        attack_weight=config.get('attack_weight',.03),defense_weight=0.,margin=.01,
        loss_temperature=.1,rival_source='base',main_objective='identity_score',
        defense_objective='shared_capture',defense_margin=.2)
    neutral=((shared*(own-other)).sum(-1)).square().mean()
    total=base+config.get('neutral_weight',.03)*neutral
    return total,dict(ce=parts['ce'],identity=parts['main'],attack=parts['attack'],neutral=neutral.detach())
