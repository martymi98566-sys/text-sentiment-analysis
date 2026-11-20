"""One English review -> original-text highlights, aspect/sign shares, local HTML/JSON/PNG.

Local research app, preserving first-board weights and extraction; optional reason policy.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime
from unittest.mock import patch

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from sentiment_analysis.reason_composition import composition,overall_inference,render,score_choices,write_json
import sentiment_analysis.reason_composition as comp
from sentiment_analysis.reason_sentence_first import collect_units,decide_local_sign,external_reason,binary_prompt,TONE_SYSTEM
from sentiment_analysis.reason_serving_guard import type_guard
import render_single_review_emotions as layout


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(record,out,reason_policy=None):
    import torch
    import sentiment_analysis.reason_precision_review as review_module
    from sentiment_analysis.reason_precision_review import SemanticEvidenceVerifier
    config=load(ROOT/'configs/reason_composition_pilot.json')
    serving_path=ROOT/'configs/reason_category_serving.json'
    if reason_policy is None:
        reason_policy=load(serving_path)['active_policy'] if serving_path.exists() else 'baseline'
    if reason_policy not in {'baseline','semantic_backfill_v2','local_contrastive_reason_v8'}:
        raise ValueError('Unsupported reason category policy')
    protected=load(ROOT/'configs/runtime_integrity.json')['protected_sha256']
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'Protected model/source changed before run'
    write_json(out/'input.json',record)
    overall=overall_inference([record])[0]
    units=collect_units([record]);write_json(out/'units.json',units)
    proposed=[];held=[]
    for u in units:
        sign,why=decide_local_sign(u)
        adaptation=bool(sign and u['local_aspect_scores'].get('adaptation:'+sign,0)>=.8)
        reason=external_reason(u['text'],u['quote'],'adaptation' if adaptation else None)
        if sign and not reason:proposed.append(dict(u,polarity=sign,polarity_source=why,category='local evaluation'))
        else:held.append(dict(u,unresolved_reason=reason or why))
    model=SemanticEvidenceVerifier(batch_size=4,streaming=True,use_prefix_cache=False)
    with patch.object(review_module,'prompt',binary_prompt):checks=model.score(proposed,'joint')
    selected=[]
    for u,s in zip(proposed,checks):
        risk=bool(':' in u['quote'] or re.search(r'\b(?:yeah right|as if|supposedly)\b',u['quote'],re.I))
        if risk and (s.get('status')!='scored' or s.get('yes_score',0)<.5):
            held.append(dict(u,unresolved_reason='contextual_irony_needs_review'));continue
        selected.append(dict(u,context_check_advisory=s,context_check_disagrees=s.get('status')!='scored' or s.get('yes_score',0)<.5))
    with patch.object(comp,'TONE_SYSTEM',TONE_SYSTEM):tones=score_choices(model,[dict(u,text=u['quote']) for u in selected],'intensity')
    emotions=[]
    for u,s in zip(selected,tones):
        if s.get('status')=='scored' and s['choice_mass']>=.35:
            emotions.append(dict(u,intensity=['mild','ordinary','strong'][s['answer']],intensity_assessment=s,reason_categories=[],human_reviewed=False,tone_calibrated=False))
        else:held.append(dict(u,unresolved_reason='tone_not_assessed'))
    categories=load(ROOT/'configs/review_application.json')['categories'];candidates=[]
    for u in emotions:
        if u['status']=='overall_attitude':continue
        local=[]
        for category in categories:
            score=u['local_aspect_scores'][category['id']+':'+u['polarity']]
            candidate=dict(u,category=category['id'],nli_score=score,candidate_id=u['unit_id']+':'+category['id'])
            ok,_=type_guard(candidate)
            if score>=.8 and ok:local.append(candidate)
        if any(c['category']!='other' for c in local):local=[c for c in local if c['category']!='other']
        candidates.extend(local)
    scores=model.score(candidates,'joint');index={u['unit_id']:u for u in emotions};evidence=[]
    for c,s in zip(candidates,scores):
        if s.get('status')=='scored' and s['yes_score']>=.5 and s.get('binary_mass',0)>=.5:
            index[c['unit_id']]['reason_categories'].append(c['category']);evidence.append(dict(c,joint=s))
    del model;gc.collect();torch.cuda.empty_cache()
    row=dict(record,overall=overall,opinion_units=units,emotion_units=emotions,evidence=evidence,
             overall_attitude_units=[u for u in emotions if u['status']=='overall_attitude'],rejected_evidence=[],
             composition=composition(record['text'],evidence,config['intensity_weights']))
    if reason_policy in {'semantic_backfill_v2','local_contrastive_reason_v8'}:
        from sentiment_analysis.reason_relaxed_final import backfill_row
        row=backfill_row(row)
        if reason_policy=='local_contrastive_reason_v8':
            from sentiment_analysis.reason_router_v8 import apply_router
            router_dir=ROOT/'data/reason_training/reason_router_v8_20261120'
            qualified=load(router_dir/'deployment_decision.json')['passed']
            trial=load(router_dir/'local_trial_release.json') if (router_dir/'local_trial_release.json').exists() else {}
            if not qualified and not trial.get('local_trial_approved'):
                raise RuntimeError('Reason policy has neither formal qualification nor a documented local trial review')
            if trial.get('local_trial_approved'):
                assert sha(router_dir/'deployment_decision.json')==trial['formal_decision_sha256']
            router_protocol=load(router_dir/'evaluation_protocol.json')
            assert all(sha(ROOT/p)==h for p,h in router_protocol['fingerprints'].items())
            assert sha(router_dir/'selected_policy.json')==router_protocol['policy_sha256']
            row=apply_router(row,load(router_dir/'selected_policy.json'))
            row['reason_release_tier']='qualified' if qualified else 'local_research_trial'
            row['reason_formal_acceptance_passed']=qualified
        write_json(out/'reason_category_decisions.json',[{'unit_id':u['unit_id'],'categories':u['reason_categories'],
                   'decision':u['reason_classification'],'type_scores':u.get('reason_type_scores')} for u in row['emotion_units']])
    else:row['reason_policy']='baseline'
    write_json(out/'polarity_verifications.json',[dict(u,check=s) for u,s in zip(proposed,checks)])
    write_json(out/'category_scores.json',[dict(c,joint=s) for c,s in zip(candidates,scores)])
    write_json(out/'unit_held.json',held);write_json(out/'results.json',[row]);render([row],out)
    layout.OUT=out;names={c['id']:c['name'] for c in categories}
    (out/'report.html').write_text(layout.page([row],names,True),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'Protected model/source changed during run'
    write_json(out/'verification.json',{'protected_count':len(protected),'protected_unchanged':True,
               'new_training':reason_policy=='local_contrastive_reason_v8','first_board_changed':False,'serving_changed':reason_policy!='baseline','reason_policy':reason_policy,'independent_evaluation':False,
               'html_source_preserved':True,'source_sha256':sha(Path(__file__)),
               'input_sha256':sha(out/'input.json'),'composition_config_sha256':sha(ROOT/'configs/reason_composition_pilot.json'),
               'context_disagreements':sum(u.get('context_check_disagrees',False) for u in emotions)})
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text');group.add_argument('--text-file',type=Path)
    parser.add_argument('--movie',default='未提供片名（用户输入，不自动推断）')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reason-policy',choices=['baseline','semantic_backfill_v2','local_contrastive_reason_v8'],default=None,
                        help='Default follows configs/reason_category_serving.json; baseline replays the original category policy')
    args=parser.parse_args()
    text=args.text if args.text is not None else args.text_file.read_text(encoding='utf-8-sig')
    if not text.strip():parser.error('评论不能为空')
    if not re.search(r'[A-Za-z]',text):parser.error('当前模型仅验证过英文评论，请输入英文原文')
    out=args.output or ROOT/'data/reason_composition/single_inputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=out.resolve()
    if out.exists() and any(out.iterdir()):parser.error('输出目录非空，请选择新目录以保留旧结果')
    out.mkdir(parents=True,exist_ok=True)
    row=analyze({'review_id':'SINGLE','movie_title':args.movie,'text':text},out,args.reason_policy)
    print(json.dumps({'output':str(out),'overall':row['overall']['overall_sentiment'],
                      'slices':row['composition']['slices'],'development_prototype':True},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
