"""Shared model factory, retained for checkpoint compatibility."""
from maft.attack_defense_views import AttackDefenseViews
from maft.score_repair import ScoreRepair

def make_model(meta, mode, adapter):
    kwargs=dict(game_scale_init=1,game_scale_max=100,concept_temperature=.1,
        view_temperature=.5,evidence_temperature=.3,negative_mode='shared',
        score_mode='exclusive',support_temperature=.5,visual_adapter_dim=adapter,
        visual_adapter_scope='all')
    if mode in ('smooth','smooth_ce'):
        return ScoreRepair(meta['class_text'],meta['concept_text'],repair='smooth',**kwargs)
    return AttackDefenseViews(meta['class_text'],meta['concept_text'],game=mode!='ng',**kwargs)
