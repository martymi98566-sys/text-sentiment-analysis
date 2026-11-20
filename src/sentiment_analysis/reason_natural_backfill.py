"""Trained natural-target adjudication after unchanged extraction and v2 categories."""
import gc,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_fragment_model import TargetReasonModel,encode_focus,collate_focus
from sentiment_analysis.reason_relaxed import CONCRETE_OTHER


def predict_natural_targets(records,directory):
    directory=Path(directory);cfg=json.loads((directory/'inference_config.json').read_text(encoding='utf-8'))
    assert cfg['input_mode']=='target_only'
    model_path=ROOT/cfg['model_path'];tok=AutoTokenizer.from_pretrained(model_path/'encoder',local_files_only=True)
    model=TargetReasonModel.load(model_path,'cuda');model.eval();torch.set_num_threads(6)
    answers=[None]*len(records);encoded=[];indices=[]
    for i,r in enumerate(records):
        quote=r['text'][r['start']:r['end']]
        if quote!=r['quote']:raise ValueError('Quote/source mismatch')
        try:encoded.append(encode_focus(tok,quote,0,len(quote),cfg['max_length']));indices.append(i)
        except ValueError:answers[i]={'category_scores':{},'specific_score':None,'status':'target_exceeds_budget'}
    with torch.inference_mode():
        for a in range(0,len(encoded),16):
            inputs={k:v.to('cuda') for k,v in collate_focus(tok,encoded[a:a+16]).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,gate=model(**inputs)
            scores=logits.float().sigmoid().cpu().tolist();gates=gate.float().sigmoid().cpu().tolist()
            for i,values,g in zip(indices[a:a+16],scores,gates):
                answers[i]={'category_scores':dict(zip(cfg['labels'],values)),'specific_score':g,'status':'scored','scores_are_calibrated_probabilities':False}
    del model;gc.collect();torch.cuda.empty_cache();return answers


def adjudicate(unit,prediction,policy):
    """Scores select labels only; they never enter intensity or pie percentages."""
    before=list(unit.get('reason_categories',[]));old=unit.get('reason_classification',{});why=old.get('reason')
    if why=='preserve_existing_supported_categories':return before,{'reason':'preserve_existing_supported_categories'}
    if prediction.get('status')!='scored':return before,{'reason':'natural_model_abstained','model_status':prediction.get('status')}
    gate=prediction['specific_score']
    if before and gate<policy['veto_gate']:
        return [],{'reason':'natural_no_specific_reason_veto','previous_categories':before,'specific_score':gate}
    if before:return before,{'reason':'preserve_semantic_v2_categories'}
    if unit['status'] in {'overall_attitude','factual','uncertain'}:return [],{'reason':'preserve_non_specific_unit_status'}
    # Respect existing domain, quotation, general-verdict and scope rejections.
    if why not in {'no_reason_with_sufficient_support','no_concrete_other_reason','audibility_without_supported_sound_label'}:
        return [],{'reason':'preserve_scope_or_generic_abstention','previous_reason':why}
    if gate<policy['gate_threshold']:return [],{'reason':'natural_specificity_below_threshold','specific_score':gate}
    types=unit.get('reason_type_scores',{});chosen=[]
    for category,score in prediction['category_scores'].items():
        if score<policy['category_threshold'] or types.get(category,0)<policy['semantic_min']:continue
        if category=='other' and not CONCRETE_OTHER.search(unit['quote']):continue
        chosen.append(category)
    return chosen,{'reason':'natural_trained_backfill' if chosen else 'natural_category_not_confirmed','specific_score':gate}


def apply_natural_backfill(row,directory,policy):
    from sentiment_analysis.reason_composition import composition
    units=row['emotion_units'];predictions=predict_natural_targets(units,directory) if units else []
    emotions=[];evidence=[]
    for u,p in zip(units,predictions):
        categories,decision=adjudicate(u,p,policy)
        new=dict(u,reason_categories=categories,reason_classification=dict(decision,previous_decision=u.get('reason_classification'),
            natural_model=p,policy=policy['name']))
        emotions.append(new)
        for c in categories:evidence.append(dict(new,category=c,nli_score=u['local_aspect_scores'].get(c+':'+u['polarity'],0)))
    weights=json.loads((ROOT/'configs/reason_composition_pilot.json').read_text(encoding='utf-8'))['intensity_weights']
    return dict(row,emotion_units=emotions,evidence=evidence,reason_policy='natural_target_backfill_v3',
        overall_attitude_units=[u for u in emotions if u['status']=='overall_attitude'],composition=composition(row['text'],evidence,weights))
