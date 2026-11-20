"""Compare reason categories after frozen emotion extraction; never invent opinion spans."""
import gc
import json
import re
from pathlib import Path

from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_precision_v4 import ANCHORS, WRITTEN_DIALOGUE
from sentiment_analysis.reason_sentence_first import external_reason
from sentiment_analysis.topic_language import generic_verdict_rule

TYPE_HYPOTHESES = {
    'plot': "This passage evaluates the quality of the movie's plot, story writing, coherence, originality or ending.",
    'pacing': "This passage evaluates how fast or slow the movie moves, its editing, rhythm or scene length.",
    'acting': "This passage evaluates the actors' performances, their delivery or casting in the movie.",
    'characters': "This passage evaluates how the fictional characters are written, developed or motivated in the movie.",
    'dialogue': "This passage evaluates the wording, wit or naturalness of the movie's written dialogue or lines.",
    'visuals': "This passage evaluates the movie's cinematography, visual effects, animation, costumes, sets or visual production.",
    'sound': "This passage evaluates the movie's music, songs, rapping, sound effects, recording, mixing or audible clarity.",
    'emotion': "This passage evaluates the movie's humor, suspense, fear, thrills, or its ability to move the audience.",
    'theme': "This passage evaluates the movie's ideas, message, meaning, depth or social expression.",
    'adaptation': "This passage evaluates how successfully the movie adapts a book or other original work.",
    'other': "This passage evaluates a concrete production or viewing issue outside the usual plot, acting, visual and sound categories.",
    'generic': "This passage gives only a general verdict or recommendation about the movie without a specific feature or reason.",
    'factual': "This passage describes events, people or story facts without evaluating movie quality.",
    'external': "This passage evaluates another review, a trailer, or a different or future work instead of the current movie.",
}

POLICIES = {
    'local_ranked': {'local_weight':.85,'semantic_weight':.15,'anchor_bonus':.08,'minimum_rank':.50,'secondary_gap':.14},
    'semantic_ranked': {'local_weight':.40,'semantic_weight':.60,'anchor_bonus':.08,'minimum_rank':.48,'secondary_gap':.18},
}
SCOPE_ONLY = re.compile(r"\b(?:previews?|trailers?|pre-production|negative\s+comment|other\s+review\w*|critics?\s+(?:said|say|think)|will\s+(?:shock|scare|amaze|be\s+nothing))\b",re.I)
CONCRETE_OTHER = re.compile(r'\b(?:budget|subtitles?|distribution|ticket|price|accessib\w*|age\s+(?:rating|suitability)|direct(?:or|ion)|advertis\w*|commercial\s+breaks?)\b',re.I)


def score_reason_types(units):
    import torch
    from sentiment_analysis.reason_repair import load_model
    from sentiment_analysis.reason_precision_v4_nli import low_memory_scores
    tok,model=load_model(ROOT/'data/pretrained/topic_nli')
    quotes=list(dict.fromkeys(u['quote'] for u in units))
    labels=list(TYPE_HYPOTHESES)
    scores=low_memory_scores(model,tok,[(q,TYPE_HYPOTHESES[c]) for q in quotes for c in labels]).reshape(-1,len(labels))
    mapping={q:dict(zip(labels,s.tolist())) for q,s in zip(quotes,scores)}
    del model;gc.collect();torch.cuda.empty_cache()
    return [dict(u,reason_type_scores=mapping[u['quote']]) for u in units]


def assign_categories(unit,policy='semantic_ranked'):
    """Ranking scores are selection signals, never intensity or percentage weights."""
    params=POLICIES[policy] if isinstance(policy,str) else policy
    types=unit['reason_type_scores'];local=unit['local_aspect_scores'];sign=unit['polarity'];quote=unit['quote']
    categories=list(TYPE_HYPOTHESES)[:11]
    if unit['status']=='overall_attitude' or generic_verdict_rule(quote):return [],{'reason':'overall_attitude'}
    if SCOPE_ONLY.search(quote):return [],{'reason':'external_or_future_evaluation'}
    adaptation=types['adaptation']>=.45 or local.get('adaptation:'+sign,0)>=.8
    scope=external_reason(unit['text'],quote,'adaptation' if adaptation else None)
    if scope:return [],{'reason':scope}
    best_specific=max(types[c] for c in categories[:-1])
    if types['generic']>=.5 and types['generic']>=best_specific+.03:return [],{'reason':'generic_without_specific_reason'}
    if types['factual']>=.75 and types['factual']>=best_specific+.10:return [],{'reason':'not_a_specific_quality_judgment'}
    ranked=[]
    for category in categories:
        score=local.get(category+':'+sign,0)
        anchor=bool(ANCHORS[category].search(quote))
        value=params['local_weight']*score+params['semantic_weight']*types[category]+params['anchor_bonus']*anchor
        ranked.append({'category':category,'rank':value,'local':score,'semantic':types[category],'anchor':anchor})
    ranked.sort(key=lambda r:(-r['rank'],r['category']))
    best=ranked[0]
    recording_issue=bool(re.search(r'\b(?:hear|audible|inaudible|recording|mixing|volume)\b',quote,re.I) and re.search(r'\b(?:dialogue|lines|voices?|sound)\b',quote,re.I) and not WRITTEN_DIALOGUE.search(quote))
    if recording_issue:
        sound=next(r for r in ranked if r['category']=='sound')
        if max(sound['local'],sound['semantic'])>=.35:return ['sound'],{'reason':'audibility_is_sound_not_written_dialogue','ranking':ranked}
        return [],{'reason':'audibility_without_supported_sound_label','ranking':ranked}
    if best['category']=='other' and not CONCRETE_OTHER.search(quote):
        specific=next((r for r in ranked if r['category']!='other' and r['local']>=.90 and r['semantic']>=.40),None)
        if specific:best=specific
    if best['category']=='other':
        if not CONCRETE_OTHER.search(quote) or types['other']<.5 or types['other']<=max(types['generic'],types['factual']):
            return [],{'reason':'no_concrete_other_reason','ranking':ranked}
    if best['rank']<params['minimum_rank'] or max(best['local'],best['semantic'])<.60:
        return [],{'reason':'no_reason_with_sufficient_support','ranking':ranked}
    chosen=[best['category']]
    if best['category']!='other':
        for row in ranked:
            if row['category']==best['category']:continue
            if row['category']=='other':continue
            # Keep genuine multi-aspect phrases; weak related categories do not all enter.
            if row['anchor'] and best['anchor'] and row['local']>=.80 and row['semantic']>=.35 and best['rank']-row['rank']<=params['secondary_gap']:
                chosen.append(row['category'])
    return chosen,{'reason':'ranked_best_fit','ranking':ranked,'policy':policy}


def apply_reason_policy(row,policy):
    from sentiment_analysis.reason_composition import composition
    config=json.loads((ROOT/'configs/reason_composition_pilot.json').read_text(encoding='utf-8'))
    emotions=[];evidence=[]
    for original in row['emotion_units']:
        if original.get('reason_categories'):
            categories=list(original['reason_categories']);decision={'reason':'preserve_existing_supported_categories'}
        else:
            categories,decision=assign_categories(original,policy)
        u=dict(original,reason_categories=categories,reason_classification=decision)
        emotions.append(u)
        for category in categories:evidence.append(dict(u,category=category,nli_score=original['local_aspect_scores'].get(category+':'+original['polarity'],0)))
    return dict(row,emotion_units=emotions,evidence=evidence,
                overall_attitude_units=[u for u in emotions if u['status']=='overall_attitude'],
                reason_policy=policy,composition=composition(row['text'],evidence,config['intensity_weights']))
