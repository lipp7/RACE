import torch
from torch import nn
from torch.nn import functional as F


def cosine(a, b):
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


class AttackDefenseViews(nn.Module):
    """Instance-conditioned class agents with visual attack/defense evidence.

    The game term is part of the classifier, so its losses cannot be satisfied
    solely by moving text vectors while leaving classification unchanged.
    """

    def __init__(self, class_text, concept_text, game=True, concept_temperature=0.05,
                 view_temperature=0.2, evidence_temperature=0.5,
                 visual_relevance_weight=0.5, game_scale_init=10.0,
                 head_prompt_weight=1.0, anchor_mode="head", game_scale_max=100.0,
                 score_mode="identity", head_init="random", head_init_scale=0.5,
                 ambiguity_temperature=0.0, prompt_mode="dynamic", use_conditioner=True,
                 semantic_scale=10.0, local_mode="attention", rival_mode="hardest",
                 opponent_count=1, concept_contrast=0.0, visual_adapter_dim=0, visual_adapter_scope="all",
                 negative_mode="opponent", support_temperature=0.0):
        super().__init__()
        class_text = F.normalize(class_text.float(), dim=-1)
        concept_text = F.normalize(concept_text.float(), dim=-1)
        classes, dim = class_text.shape
        concepts = concept_text.shape[0]
        self.register_buffer("class_text", class_text)
        self.register_buffer("concept_text", concept_text)
        self.register_buffer("concept_prior", class_text @ concept_text.T / concept_temperature)
        self.head = nn.Linear(dim, classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if head_init == "text":
            with torch.no_grad():
                self.head.weight.copy_(head_init_scale * class_text)
        elif head_init != "random":
            raise ValueError("Unknown head initialization")
        self.ambiguity_temperature = float(ambiguity_temperature)
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.concept_delta = nn.Parameter(torch.zeros(classes, concepts))
        self.prompt_delta = nn.Parameter(torch.zeros(classes, dim))
        self.conditioner = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(),
                                         nn.Linear(dim // 2, dim))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.game_scale = nn.Parameter(torch.tensor(float(game_scale_init)))
        self.game_scale_max = float(game_scale_max)
        if score_mode not in ("identity", "competitive", "exclusive"):
            raise ValueError("Unknown score_mode")
        self.score_mode = score_mode
        self.game_progress = 1.0
        self.head_prompt_weight = nn.Parameter(torch.tensor(float(head_prompt_weight)))
        self.game = game
        self.concept_temperature = concept_temperature
        self.view_temperature = view_temperature
        self.evidence_temperature = evidence_temperature
        self.visual_relevance_weight = visual_relevance_weight
        if prompt_mode not in ("dynamic", "static") or local_mode not in ("attention", "mean", "none"):
            raise ValueError("Invalid prompt or local mode")
        if rival_mode not in ("hardest", "fixed"):
            raise ValueError("Invalid rival mode")
        if local_mode == "none" and game:
            raise ValueError("Global-only ablation must also disable local game score")
        self.prompt_mode, self.use_conditioner = prompt_mode, bool(use_conditioner)
        self.opponent_count=min(int(opponent_count),classes-1)
        self.concept_contrast=float(concept_contrast)
        similarity=class_text@ class_text.T
        similarity.fill_diagonal_(-torch.inf)
        self.semantic_neighbors=similarity.topk(min(3,classes-1),dim=-1).indices
        self.semantic_scale, self.local_mode, self.rival_mode = float(semantic_scale), local_mode, rival_mode
        generator = torch.Generator().manual_seed(2026)
        table = torch.randint(classes - 1, (classes,), generator=generator)
        self.fixed_rivals = table + (table >= torch.arange(classes)).long()
        if anchor_mode not in ("head", "prompt", "fusion", "text"):
            raise ValueError("anchor_mode must be head, prompt, or fusion")
        self.anchor_mode = anchor_mode
        if visual_adapter_scope not in ('all','local'):raise ValueError('Invalid adapter scope')
        self.visual_adapter_scope=visual_adapter_scope
        if negative_mode not in ('opponent','shared'):raise ValueError('Invalid negative evidence mode')
        self.negative_mode=negative_mode
        self.support_temperature=float(support_temperature)
        self.visual_adapter=(nn.Sequential(nn.Linear(dim,visual_adapter_dim),nn.ReLU(),nn.Linear(visual_adapter_dim,dim))
                             if visual_adapter_dim else None)
        if self.visual_adapter is not None:
            nn.init.zeros_(self.visual_adapter[-1].weight);nn.init.zeros_(self.visual_adapter[-1].bias)

    def load_view_pretrain(self, state):
        """Initialize both game and no-game variants from the same view model."""
        with torch.no_grad():
            self.head.weight.copy_(state["head.weight"])
            self.head.bias.copy_(state["head.bias"])
            self.beta.copy_(state["beta"])
            self.concept_delta.copy_(state["concept_delta"])

    def forward(self, views):
        views = F.normalize(views.float(), dim=-1)
        feature_drift=None
        if self.visual_adapter is not None:
            original=views
            if self.visual_adapter_scope=='local':
                local=F.normalize(views[:,1:]+self.visual_adapter(views[:,1:]),dim=-1)
                views=torch.cat([views[:,:1],local],dim=1)
            else:views=F.normalize(views+self.visual_adapter(views),dim=-1)
            feature_drift=(1-(views*original).sum(-1)).mean()
        global_view, local_views = views[:, 0], views[:, 1:]
        batch = views.shape[0]
        visual_prototype = F.normalize(self.head.weight, dim=-1)
        conditioning = (self.conditioner(global_view)[:, None]
                        if self.prompt_mode == "dynamic" and self.use_conditioner else
                        torch.zeros(batch, 1, global_view.shape[-1], device=views.device))
        prompt = F.normalize(self.class_text[None] + self.head_prompt_weight * visual_prototype[None] +
                             self.prompt_delta[None] + conditioning, dim=-1)
        if self.anchor_mode == "head":
            anchor = visual_prototype[None].expand(batch, -1, -1)
        elif self.anchor_mode == "fusion":
            anchor = F.normalize(prompt + visual_prototype[None], dim=-1)
        elif self.anchor_mode == 'text':
            anchor=self.class_text[None].expand(batch,-1,-1)
        else:
            anchor = prompt
        responses = local_views @ self.concept_text.T
        image_relevance = (responses.max(dim=1).values if self.prompt_mode == "dynamic" else
                           torch.zeros(batch, responses.shape[-1], device=views.device))
        semantic_scores=torch.einsum("bcd,kd->bck", prompt, self.concept_text)
        if self.concept_contrast:
            neighbors=self.semantic_neighbors.to(views.device)
            contrast=semantic_scores-semantic_scores[:,neighbors].mean(2)
            semantic_scores=semantic_scores+self.concept_contrast*contrast
        concept_logits = (semantic_scores +
                          self.visual_relevance_weight * image_relevance[:, None]) / self.concept_temperature
        concept_logits = concept_logits + self.concept_delta[None]
        concept_weights = torch.softmax(concept_logits, dim=-1)
        concept_match = torch.einsum("brk,bck->bcr", responses, concept_weights)
        local_scores = self.head(local_views).transpose(1, 2)
        gate = local_scores + self.semantic_scale * concept_match
        attention = torch.softmax(gate / self.view_temperature, dim=-1)
        local_evidence = (attention * local_scores).sum(dim=-1)
        if self.local_mode == "mean":
            local_evidence = local_scores.mean(dim=-1)
        elif self.local_mode == "none":
            local_evidence = torch.zeros_like(local_evidence)
        base_logits = self.head(global_view) + self.beta * local_evidence
        output = {"base_logits": base_logits, "prompt": prompt, "anchor": anchor,
                  "class_text":self.class_text,
                  "feature_drift":feature_drift,
                  "negative_mode":self.negative_mode,
                  "concept_weights": concept_weights, "attention": attention,
                  "gate": gate, "local_views": local_views,
                  "evidence_temperature": self.evidence_temperature}
        if not self.game:
            output["logits"] = base_logits
            return output
        # Each candidate class selects its strongest competing agent from
        # global + local evidence, without differentiating through argmax.
        rivals = base_logits.detach().unsqueeze(1).expand(-1, base_logits.shape[1], -1).clone()
        rivals.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
        rival_ids = (rivals.argmax(dim=-1) if self.rival_mode == "hardest" else
                     self.fixed_rivals.to(views.device)[None].expand(batch, -1))
        rival_gate = gate.gather(1, rival_ids[:, :, None].expand_as(gate))
        opponent_anchor=None
        if self.opponent_count>1 and self.rival_mode=='hardest':
            values, ids=rivals.topk(self.opponent_count,dim=-1)
            mixture=torch.softmax(values,dim=-1)
            batch_ids=torch.arange(batch,device=views.device)[:,None,None]
            rival_gate=(gate[batch_ids,ids]*mixture[...,None]).sum(2)
            opponent_anchor=F.normalize((anchor[batch_ids,ids]*mixture[...,None]).sum(2),dim=-1)
            output.update({'opponent_gate':rival_gate,'opponent_anchor':opponent_anchor,'opponent_count':self.opponent_count})
        difference = gate - rival_gate
        plus_weights = torch.softmax(difference / self.evidence_temperature, dim=-1)
        negative_gate=(torch.minimum(gate,rival_gate) if self.negative_mode=='shared' else -difference)
        minus_weights = torch.softmax(negative_gate / self.evidence_temperature, dim=-1)
        plus = F.normalize(torch.einsum("bcr,brd->bcd", plus_weights, local_views), dim=-1)
        minus = F.normalize(torch.einsum("bcr,brd->bcd", minus_weights, local_views), dim=-1)
        identity = cosine(plus, anchor) - cosine(minus, anchor)
        if self.score_mode == "competitive":
            rival_anchor = (opponent_anchor if opponent_anchor is not None else
                            anchor.gather(1, rival_ids[:, :, None].expand_as(anchor)))
            # Compare own positive evidence with the rival's positive evidence.
            # For mutual rivals this score is antisymmetric, unlike summing the
            # attack/defense advantages, which boosts both mutual rivals equally.
            identity = cosine(plus, anchor) - cosine(minus, rival_anchor)
        elif self.score_mode == 'exclusive':
            rival_anchor=(opponent_anchor if opponent_anchor is not None else anchor.gather(1,rival_ids[:,:,None].expand_as(anchor)))
            direction=F.normalize(anchor-rival_anchor,dim=-1)
            # Keep the residual magnitude: a candidate with no private evidence
            # should not become confident merely through normalization.
            identity=((plus-minus)*direction).sum(-1)
            if self.support_temperature>0:
                support=torch.sigmoid(difference/self.support_temperature).mean(-1)
                identity=identity*support;output['support_mass']=support
        score_adjustment = identity
        if self.ambiguity_temperature > 0:
            rival_score = base_logits.gather(1, rival_ids)
            preference = torch.sigmoid((base_logits - rival_score).detach() / self.ambiguity_temperature)
            ambiguity = 4 * preference * (1 - preference)
            score_adjustment = ambiguity * identity
            output["game_ambiguity"] = ambiguity
        output.update({"logits": base_logits + self.game_progress *
                       self.game_scale.clamp(0, self.game_scale_max) * score_adjustment,
                       "plus": plus, "minus": minus, "identity": identity,
                       "rival_ids": rival_ids, "rival_mode": self.rival_mode,
                       "fixed_rivals": self.fixed_rivals.to(views.device)})
        return output


def attack_defense_loss(output, labels, main_weight=0.1, attack_weight=0.1,
                        defense_weight=0.1, margin=0.05, loss_type="softplus",
                        loss_temperature=1.0, rival_source="final", auxiliary_weighting="none",
                        main_objective="sum", separation_weight=0.0, semantic_anchor_weight=0.0,
                        defense_objective="rival",defense_margin=.2):
    logits = output["logits"]
    batch = labels.shape[0]
    rows = torch.arange(batch, device=labels.device)
    rivals = output["base_logits" if rival_source == "base" else "logits"].detach().clone()
    rivals[rows, labels] = -torch.inf
    rival_ids = rivals.argmax(-1)
    if output.get("rival_mode", "hardest") == "fixed":
        rival_ids = output["fixed_rivals"][labels]
    gate, views, anchor = output["gate"], output["local_views"], output["anchor"]
    ga, gb = gate[rows, labels], gate[rows, rival_ids]
    if output.get('opponent_count',1)>1:
        if rival_source!='base':raise ValueError('Multi-opponent losses must use the same base-score rivals as inference')
        gb=output['opponent_gate'][rows,labels]
    difference = ga - gb
    temperature = output.get("evidence_temperature", 0.5)
    plus = F.normalize(torch.einsum("br,brd->bd", torch.softmax(difference / temperature, dim=-1), views), dim=-1)
    negative_gate=(torch.minimum(ga,gb) if output.get('negative_mode','opponent')=='shared' else -difference)
    minus = F.normalize(torch.einsum("br,brd->bd", torch.softmax(negative_gate / temperature, dim=-1), views), dim=-1)
    qa, qb = anchor[rows, labels], anchor[rows, rival_ids]
    if output.get('opponent_count',1)>1:qb=output['opponent_anchor'][rows,labels]
    attack_advantage = cosine(plus, qa) - cosine(plus, qb)
    defense_advantage = cosine(minus, qb) - cosine(minus, qa)
    if auxiliary_weighting == "uncertainty":
        sample_weight = 1 - logits.detach().softmax(-1)[rows, labels]
    elif auxiliary_weighting == "none":
        sample_weight = None
    else:
        raise ValueError("Unknown auxiliary weighting")
    # The original auxiliary comparison uses one shared pair of evidence.
    # Its attack+defense sum differs from the candidates' own identity-score gap.
    def penalty(advantage,target=None):
        violation = (margin if target is None else target) - advantage
        if loss_type == "hinge":
            terms = F.relu(violation)
        elif loss_type == "softplus" and loss_temperature > 0:
            terms = loss_temperature * F.softplus(violation / loss_temperature)
        else:
            raise ValueError("Invalid attack/defense penalty configuration")
        return terms.mean() if sample_weight is None else (sample_weight * terms).mean()
    main_gap=(output['identity'][rows,labels]-output['identity'][rows,rival_ids]
              if main_objective=='identity_score' else attack_advantage+defense_advantage)
    main = penalty(main_gap)
    attack = penalty(attack_advantage)
    defense = (penalty((cosine(minus,qa)+cosine(minus,qb))/2,defense_margin)
               if defense_objective=='shared_capture' else penalty(defense_advantage))
    ce = F.cross_entropy(logits, labels)
    total = ce + main_weight * main + attack_weight * attack + defense_weight * defense
    separation=F.relu(cosine(plus,minus)-.85).mean()
    if separation_weight:total=total+separation_weight*separation
    if semantic_anchor_weight:
        tether=1-cosine(output['prompt'],output['class_text'][None]).mean()
        total=total+semantic_anchor_weight*tether
    parts = {"ce": ce.detach(), "main": main.detach(),
             "attack": attack.detach(), "defense": defense.detach()}
    if sample_weight is not None:
        parts["auxiliary_weight"] = sample_weight.mean().detach()
    return total, parts
