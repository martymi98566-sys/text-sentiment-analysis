"""Learned specificity + semantic categories + shared category-boundary checks."""
import json,re
from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_precision_v4 import ANCHORS
from sentiment_analysis.reason_relaxed import CONCRETE_OTHER
from sentiment_analysis.reason_precision_anchor import EXCLUDED_MEDIUM
from sentiment_analysis.reason_natural_backfill import predict_natural_targets
from sentiment_analysis.reason_semantic_router_v5 import score_routes,LABELS

APPEARANCE=re.compile(r'\b(?:eye[ -]candy|photogenic|roving\s+eye|sex\s+appeal|sexy|gorgeous|handsome|attractive|babe|looks?\s+(?:\w+\s+){0,3}(?:fantastic|beautiful|stunning|hot|good))\b',re.I)
PERFORMANCE=re.compile(r'\b(?:acting|perform\w*|portray\w*|delivery|miscast|casting|(?:great|good|terrific|poor|wooden)\s+as)\b',re.I)
VISUAL_CUES=re.compile(ANCHORS['visuals'].pattern+r'|\b(?:shots?|images?|imagery|framing|palette)\b',re.I)
ADAPTATION_CUES=re.compile(r'\b(?:adapt\w*|remak\w*|remade|faithful\w*|unfaithful\w*|novels?|books?|source|original\s+(?:play|version))\b',re.I)
PACING_CUES=re.compile(ANCHORS['pacing'].pattern+r'|\b(?:stretch\w*|expand\w*|padding|running\s+time|runtime|length|scene\s+changes)\b',re.I)
CHAR_CUES=re.compile(ANCHORS['characters'].pattern+r'|\b(?:depth|arcs?|growth|develop\w*|motivat\w*|believ\w*|credib\w*|convinc\w*|feel\w*\s+real|never\s+chang\w*)\b',re.I)


def category_supported(unit,category,semantic_min):
    quote=unit['quote'];types=unit.get('reason_type_scores',{})
    if category=='acting' and APPEARANCE.search(quote) and not PERFORMANCE.search(quote):return False
    if category=='visuals' and not VISUAL_CUES.search(quote):return False
    if category=='adaptation' and not ADAPTATION_CUES.search(quote):return False
    if category=='other' and not (CONCRETE_OTHER.search(quote) or re.search(r'\b(?:title|dubbing|captions?|cost|worth)\b',quote,re.I)):return False
    cue=PACING_CUES if category=='pacing' else CHAR_CUES if category=='characters' else ANCHORS[category]
    return bool(cue.search(quote) or types.get(category,0)>=semantic_min)


def route_categories(unit,prediction,policy):
    before=list(unit.get('reason_categories',[]));previous=unit.get('reason_classification',{})
    if previous.get('reason')=='preserve_existing_supported_categories':return before,{'reason':'preserve_existing_supported_categories'}
    route=prediction['semantic_route'];specificity=prediction['learned_specificity']
    if route['status']!='scored' or specificity['status']!='scored':return before,{'reason':'consensus_abstained_preserve_previous'}
    if unit['status'] in {'overall_attitude','factual','uncertain'} or EXCLUDED_MEDIUM.search(unit['text']):return [],{'reason':'preserve_non_specific_or_excluded_status'}
    gate=specificity['specific_score']
    if gate<policy['veto_gate']:return [],{'reason':'learned_no_specific_reason','specific_score':gate}
    proposed=list(before)
    if gate>=policy['backfill_gate']:proposed.extend(route['categories'])
    categories=[c for c in LABELS if c in proposed and category_supported(unit,c,policy['semantic_min'])]
    return categories,{'reason':'trained_specificity_semantic_consensus','specific_score':gate,'previous_categories':before,
        'semantic_categories':route['categories'],'blocked_categories':sorted(set(proposed)-set(categories))}


def score_consensus(rows,semantic_routes=None):
    routes=semantic_routes if semantic_routes is not None else score_routes(rows)
    learned=predict_natural_targets(rows,ROOT/'data/reason_training/target_fragments_v3_20261120')
    return [{'semantic_route':r,'learned_specificity':g} for r,g in zip(routes,learned)]


def apply_consensus(row,policy):
    from sentiment_analysis.reason_composition import composition
    pending=[u for u in row['emotion_units'] if u.get('reason_classification',{}).get('reason')!='preserve_existing_supported_categories'
        and u['status']=='specific_evaluation']
    try:predictions=score_consensus(pending) if pending else []
    except (OSError,RuntimeError) as error:return dict(row,reason_router_fallback={'reason':'local_consensus_unavailable','error_type':type(error).__name__})
    index={u['unit_id']:p for u,p in zip(pending,predictions)};units=[];evidence=[]
    for u in row['emotion_units']:
        if u['unit_id'] in index:cats,decision=route_categories(u,index[u['unit_id']],policy)
        else:cats=list(u['reason_categories']);decision={'reason':'preserve_previously_supported_or_non_specific_unit'}
        new=dict(u,reason_categories=cats,reason_classification=dict(decision,previous_decision=u.get('reason_classification'),policy=policy['name']))
        units.append(new)
        for c in cats:evidence.append(dict(new,category=c,nli_score=u['local_aspect_scores'].get(c+':'+u['polarity'],0)))
    weights=json.loads((ROOT/'configs/reason_composition_pilot.json').read_text(encoding='utf-8'))['intensity_weights']
    return dict(row,emotion_units=units,evidence=evidence,reason_policy=policy['name'],
        overall_attitude_units=[u for u in units if u['status']=='overall_attitude'],composition=composition(row['text'],evidence,weights))
