"""Category backfill with generic-verdict and writing/performance boundaries.

Extraction, local polarity, intensity, and existing supported categories are immutable.
"""
import json
import re
from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_relaxed import assign_categories, score_reason_types

INTERJECTION=re.compile(r"^(?:ouch|oh|ugh|wow|huh|horrible|terrible|horrid|awful|great|brilliant|wonderful|bad|good|sorry|no|yes|wrong|absolutely|really)[.!?\s]*$",re.I)
GENERAL_CINEMA=re.compile(r'\b(?:masterful|masterpiece|great|excellent|amazing|wonderful|beautiful|brilliant|awful|terrible|horrible|bad|good)\s+(?:piece\s+of\s+cinema|piece\s+of\s+work|movie|film|picture)\b',re.I)
FEATURE_WORDS=re.compile(r'\b(?:plot|story|screenplay|script|acting|acted|performance|characters?|dialogue|lines|music|sound|score|editing|pacing|animation|costumes?|cinematography|effects|adaptation|theme|message|subtitles?)\b',re.I)
WRITING=re.compile(r'\b(?:badly|poorly|well|brilliantly|wonderfully|terribly|clumsily)[ -]+written\b',re.I)
SPEECH=re.compile(r'\b(?:dialogue|dialog|lines|banter|conversations?|speeches|wording|spoken|talk\w*)\b',re.I)
PERFORMANCE=re.compile(r'\b(?:poorly|badly|well|wonderfully|brilliantly|terribly)[ -]+acted\b',re.I)


def assign_final(unit):
    quote=unit['quote']
    if INTERJECTION.fullmatch(quote.strip()):return [],{'reason':'interjection_without_specific_reason'}
    if GENERAL_CINEMA.search(quote) and not FEATURE_WORDS.search(quote):return [],{'reason':'general_movie_verdict_without_reason'}
    labels,decision=assign_categories(unit,'semantic_ranked')
    if decision['reason'] not in {'ranked_best_fit','no_concrete_other_reason','no_reason_with_sufficient_support'}:
        return labels,decision
    sign=unit['polarity'];scores=unit['local_aspect_scores']
    explicit=[]
    if WRITING.search(quote) and not SPEECH.search(quote) and scores.get('plot:'+sign,0)>=.6:
        labels=[c for c in labels if c!='dialogue'];explicit.append('plot')
    if PERFORMANCE.search(quote) and scores.get('acting:'+sign,0)>=.6:explicit.append('acting')
    labels=list(dict.fromkeys(labels+explicit))
    if explicit:decision=dict(decision,reason='ranked_with_supported_writing_performance',explicit_supported=explicit)
    return labels,decision


def apply_final_policy(row):
    from sentiment_analysis.reason_composition import composition
    weights=json.loads((ROOT/'configs/reason_composition_pilot.json').read_text(encoding='utf-8'))['intensity_weights']
    emotions=[];evidence=[]
    for original in row['emotion_units']:
        if original.get('reason_categories'):
            categories=list(original['reason_categories']);decision={'reason':'preserve_existing_supported_categories'}
        else:categories,decision=assign_final(original)
        unit=dict(original,reason_categories=categories,reason_classification=decision)
        emotions.append(unit)
        for category in categories:evidence.append(dict(unit,category=category,nli_score=unit['local_aspect_scores'].get(category+':'+unit['polarity'],0)))
    return dict(row,emotion_units=emotions,evidence=evidence,reason_policy='semantic_backfill_v2',
                overall_attitude_units=[u for u in emotions if u['status']=='overall_attitude'],
                composition=composition(row['text'],evidence,weights))


def backfill_row(row):
    pending=[u for u in row['emotion_units'] if not u.get('reason_categories')]
    scored=score_reason_types(pending) if pending else []
    index={u['unit_id']:u for u in scored}
    return apply_final_policy(dict(row,emotion_units=[index.get(u['unit_id'],u) for u in row['emotion_units']]))
