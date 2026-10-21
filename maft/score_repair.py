"""Versioned scoring repairs; legacy model/checkpoints are untouched."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from maft.attack_defense_views import AttackDefenseViews, attack_defense_loss


class ScoreRepair(AttackDefenseViews):
    def __init__(self, class_text, concept_text, repair='smooth', **kwargs):
        super().__init__(class_text, concept_text, **kwargs)
        self.repair=repair
        initial=float(kwargs.get('game_scale_init',1.))
        self.scale_logit=nn.Parameter(torch.tensor(math.log(initial/(self.game_scale_max-initial))))
        self.game_scale.requires_grad_(False)
        self.register_buffer('calibration_mean',torch.zeros(len(class_text)))
        self.register_buffer('calibration_std',torch.ones(len(class_text)))

    @torch.no_grad()
    def calibrate(self, views):
        sums=torch.zeros_like(self.calibration_mean);squares=sums.clone();n=0
        for batch in views.split(128):
            local=F.normalize(batch[:,1:].to(self.class_text.device).float(),dim=-1)
            values=local@self.class_text.T
            sums+=values.sum((0,1));squares+=values.square().sum((0,1));n+=values.shape[0]*values.shape[1]
        self.calibration_mean.copy_(sums/n)
        self.calibration_std.copy_((squares/n-(sums/n).square()).clamp_min(1e-6).sqrt())

    def forward(self, views):
        out=super().forward(views)
        rows=torch.arange(len(views),device=views.device)[:,None]
        rivals=out['rival_ids'];anchor=out['anchor'];other=anchor[rows,rivals]
        identity=out['identity']
        if self.repair!='smooth':
            local=out['local_views'];gate=out['gate'];gap=gate-gate[rows,rivals]
            raw=(local@self.class_text.T).transpose(1,2)
            presence=torch.sigmoid(((raw-self.calibration_mean[None,:,None])/self.calibration_std[None,:,None]-.5)/.5)
            other_presence=presence[rows,rivals]
            positive_weights=torch.softmax(gap/self.evidence_temperature,-1)*presence*(1-other_presence)
            rival_weights=torch.softmax(-gap/self.evidence_temperature,-1)*other_presence*(1-presence)
            common_weights=presence*other_presence/local.shape[1]
            positive=positive_weights@local;negative=common_weights@local;rival_positive=rival_weights@local
            if self.repair=='null_pair':
                identity=(positive*anchor).sum(-1)-(rival_positive*other).sum(-1)-(negative*(anchor-other)).sum(-1)
            elif self.repair=='null_private':
                identity=((positive-negative)*F.normalize(anchor-other,dim=-1)).sum(-1)
            else:raise ValueError(self.repair)
            out.update(plus=positive,minus=negative,rival_positive=rival_positive,
                       evidence_mass=positive_weights.sum(-1),shared_mass=common_weights.sum(-1))
        scale=self.game_scale_max*self.scale_logit.sigmoid()
        out.update(identity=identity,effective_scale=scale,
                   logits=out['base_logits']+self.game_progress*scale*identity)
        return out


def repaired_loss(out,labels,repair):
    if repair=='smooth':
        return attack_defense_loss(out,labels,main_weight=.3,attack_weight=.03,defense_weight=.03,
                                   margin=.01,loss_temperature=.1,rival_source='base',
                                   main_objective='identity_score',defense_objective='shared_capture',defense_margin=.2)
    rows=torch.arange(len(labels),device=labels.device)
    rival=out['base_logits'].detach().clone();rival[rows,labels]=-torch.inf;rival=rival.argmax(-1)
    own=out['anchor'][rows,labels];other=out['anchor'][rows,rival]
    gap=out['logits'][rows,labels]-out['logits'][rows,rival]
    attack=(out['plus'][rows,labels]*(own-other)).sum(-1)
    shared=(out['minus'][rows,labels]*(own-other)).sum(-1)
    ce=F.cross_entropy(out['logits'],labels)
    main=(.1*F.softplus((.01-gap)/.1)).mean()
    attack_loss=(.1*F.softplus((.01-attack)/.1)).mean()
    defense=shared.square().mean()
    return ce+.3*main+.03*attack_loss+.03*defense,dict(ce=ce.detach(),main=main.detach(),attack=attack_loss.detach(),defense=defense.detach())
