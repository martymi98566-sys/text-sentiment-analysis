"""Verify candidate evidence before choosing winners or imposing the type cap."""
import re,math
from collections import defaultdict
from sentiment_analysis.reason_precision_anchor import evidence_anchor_guard,ANCHORS as V3_ANCHORS,QUALITY as V3_QUALITY,JOB

CAREER_OTHER=re.compile(r"\b(?:have|had|'ve)\s+seen\s+(?:more|better|worse|other|earlier|previous)\b|\b(?:other|previous|earlier|past)\s+(?:roles|performances)\b|\b(?:liked|loved|enjoyed|remember)\s+(?:his|her|their)\s+(?:performance|acting)\s+in\s+(?!this\b|the\s+current\b)",re.I)
WRITTEN_DIALOGUE=re.compile(r'\b(?:witty|stilted|natural|unnatural|cheesy|inane|banal|cliched|clich[eé]d|well-written|badly-written|poorly-written|wording|phrasing)\b',re.I)
MOVED_LINES=re.compile(r'\b(?:cut\w*|misplac\w*|rearrang\w*|mov\w*|omitt\w*)\b.{0,55}\b(?:lines|dialogue)\b|\b(?:lines|dialogue)\b.{0,55}\b(?:cut\w*|misplac\w*|rearrang\w*|mov\w*|omitt\w*)\b',re.I)
CHAR_WRITING=re.compile(r'\b(?:writ\w*|script|screenplay|develop\w*|underdevelop\w*|motives|motivations|relationships|one-dimensional|stereotyp\w*|caricatures?|flat\s+characters?)\b',re.I)
PACING=r'plod\w*|meander\w*|rambl\w*|bloated|overlong|long-winded|drawn\s+out|slow[ -]+(?:moving|start\w*)|(?:runs?|running)\s+(?:far\s+)?too\s+long'
EFFECT=r'funnier|funniest|amusing|amused|hysterical|chilling|heartwarming|heartbreaking|emotionally\s+affecting'
QUALITY=re.compile(V3_QUALITY.pattern+r'|\b(?:unresolved|horrid|tiresome|dreary|'+PACING+'|'+EFFECT+r')\b',re.I)
ANCHORS={**V3_ANCHORS,
 'pacing':re.compile(V3_ANCHORS['pacing'].pattern+r'|\b(?:'+PACING+r')\b',re.I),
 'characters':re.compile(V3_ANCHORS['characters'].pattern+r'|\b(?:well-developed|poorly-developed|undeveloped|underdeveloped|developed\s+characters?)\b',re.I),
 'visuals':re.compile(V3_ANCHORS['visuals'].pattern+r'|\b(?:production\s+design|art\s+direction|colou?r\w*|landscapes?)\b',re.I),
 'sound':re.compile(V3_ANCHORS['sound'].pattern+r'|\b(?:tunes?|melod\w*|acoustics)\b',re.I),
 'emotion':re.compile(V3_ANCHORS['emotion'].pattern+r'|\b(?:'+EFFECT+r')\b',re.I)}

def guard(row):
    _,reasons=evidence_anchor_guard(row);reasons=list(reasons);quote=row['quote'];category=row['category']
    clauses=re.split(r'[.;!?\n]|,|\b(?:and|but|while|whereas)\s+(?=(?:the|this|its|their|his|her|a|an)\s)',quote,flags=re.I)
    # Genre nouns and an idiom about nightmares do not by themselves evaluate
    # an aspect. Keep the exact source quote; sanitize only the lexical test.
    def quality(c):
        cleaned=re.sub(r'\b(?:thrillers?|suspense|comedy|comedies|horror)\b|\bworst\s+nightmares?\b',' ',c,flags=re.I)
        return QUALITY.search(cleaned) or (category=='acting' and re.search(JOB,cleaned,re.I))
    reasons=[r for r in reasons if r!='no_clause_with_both_explicit_aspect_and_quality']
    if not any(ANCHORS[category].search(c) and quality(c) for c in clauses):reasons.append('no_clause_with_both_explicit_aspect_and_quality')
    if category=='emotion' and re.search(r'\b(?:'+EFFECT+r')\b',quote,re.I):reasons=[r for r in reasons if r!='no_explicit_viewer_or_genre_effect']
    if CAREER_OTHER.search(quote):reasons.append('quotation_focuses_on_other_or_prior_work')
    if category=='dialogue' and MOVED_LINES.search(quote) and not WRITTEN_DIALOGUE.search(quote):reasons.append('line_placement_or_omission_not_wording_quality')
    if category=='characters' and not CHAR_WRITING.search(quote):reasons.append('no_explicit_character_writing_evidence')
    if category=='characters' and not re.search(r'\b(?:writ\w*|script|screenplay|develop\w*|underdevelop\w*|motives|motivations|relationships|one-dimensional|flat\s+characters?)\b',quote,re.I):
        reasons.append('character_stereotype_or_performer_label_without_explicit_writing')
    if category=='characters' and re.search(r'\b(?:actors?|actress\w*|performance\w*|portray\w*)\b',quote,re.I) and not re.search(r'\b(?:writ\w*|script|screenplay|develop\w*)\b',quote,re.I):
        reasons.append('performance_not_explicit_character_writing')
    return not reasons,reasons

def verification_candidates(rows,minimum_nli=.8,per_category=2):
    """Bound compute after evidence screening; never let a rejected winner block a valid quote."""
    groups=defaultdict(list)
    for r in rows:
        ok,_=guard(r)
        if ok and math.isfinite(r['nli_score']) and r['nli_score']>=minimum_nli:groups[(r['review_id'],r['category'])].append(r)
    return [r for rs in groups.values() for r in sorted(rs,key=lambda x:(-x['nli_score'],x['start'],x['end']))[:per_category]]

def select(rows,policy):
    by_review=defaultdict(dict)
    for r in rows:
        if not guard(r)[0] or not math.isfinite(r['nli_score']) or r['nli_score']<policy['nli_min']:continue
        v=r.get('joint',{})
        if v.get('status')!='scored' or not all(math.isfinite(v.get(k,float('nan'))) for k in ['yes_score','binary_mass']):continue
        if v['yes_score']<policy['joint_min'] or v['binary_mass']<policy['binary_mass_min']:continue
        rank=min(r['nli_score'],v['yes_score']);prior=by_review[r['review_id']].get(r['category'])
        if prior is None or rank>prior['selection_score']:by_review[r['review_id']][r['category']]={**r,'selection_score':rank}
    return {rid:sorted(cs.values(),key=lambda r:(-r['selection_score'],r['category']))[:policy['max_categories']] for rid,cs in by_review.items()}
