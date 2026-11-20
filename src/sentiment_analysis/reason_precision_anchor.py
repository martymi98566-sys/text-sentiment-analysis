"""Explicit evidence anchors without rejecting every comparison in a review."""
import re
from sentiment_analysis.reason_precision_guard import ANCHORS as BASE_ANCHORS,QUALITY as BASE_QUALITY

QUALITY=re.compile(BASE_QUALITY.pattern+r'|\b(?:thoughtful\w*|thoughtless\w*|lazy|lazily|cheesy|stupid|misplaced|absurd\w*|abrupt\w*|implausib\w*|plausib\w*|contrived|error\w*|inconsisten\w*|flaw\w*|pointless|confus\w*|fascinat\w*|compell\w*|captivat\w*|tedious|dreadful|pathetic|atrocious|amateur\w*|artificial\w*|clumsy|childish|pretentious|unoriginal|unrealistic|realistic|believable|unbelievable|stylish|exquisite|stunning|spectacular|imaginative|inventive|innovative|creative|delightful|delicious|charming|charismatic|powerful|riveting|memorable|forgettab\w*|forgettable|superior|inferior|better|worse|fantastic|terrific|perfection|masterful\w*|subtle|subtly|overact\w*|underact\w*|overwrought|overblown|overlong|lifeless|bland|obnoxious|annoying|irritating|fabulous|decent|magnificent|crap\w*|suck\w*)\b',re.I)
JOB=r'(?:does|did|do|doing|done)\s+(?:a\s+|an\s+)?(?:really\s+|very\s+|quite\s+)?(?:good|great|fine|excellent|fantastic|wonderful|bad|poor)\s+job'
ANCHORS={**BASE_ANCHORS,
 'acting':re.compile(r'\b(?:acting|actors?|actress\w*|perform\w*|cast\w*|portray\w*|play(?:s|ed|ing)|'+JOB+r'|(?:great|excellent|wonderful|superb|good|fine|terrific|fantastic|brilliant|convincing|perfect|poor|bad|wooden|awful|terrible)\s+as)\b',re.I),
 'pacing':re.compile(BASE_ANCHORS['pacing'].pattern+r'|\b(?:continuity|transitions?|stopped\s+as\s+abruptly)\b',re.I),
 'characters':re.compile(BASE_ANCHORS['characters'].pattern+r'|\b(?:underdeveloped|one-dimensional|stereotyp\w*|caricatures?|flat\s+characters?|characters?\s+(?:are|were|is|was)\s+(?:flat|boring|dull))\b',re.I)}
EMOTION_EFFECT=re.compile(r'\b(?:funny|unfunny|hilarious|scary|frighten\w*|terrify\w*|suspense\w*|thrill\w*|touching|moving|moved|creepy|no\s+laughs|not\s+funny|(?:I|we|you|audience|viewers?)\b.{0,25}\b(?:laugh\w*|cry\w*|scared)|humou?r\w*|jokes)\b',re.I)

EXCLUDED_MEDIUM=re.compile(r'\b(?:concert|video\s*game|videogame|gameplay|WWE|WWF|wrestling\s+(?:event|match))\b',re.I)
EXTERNAL_QUOTE=re.compile(r"\b(?:(?:the|that)\s+(?:first|original|previous|earlier|other|second)\s+(?:film|movie|version)|(?:original|previous|earlier)\s+(?:film|movie)|the\s+(?:novel|book|stage\s+play)|(?:compared?|unlike|versus))\b",re.I)

def evidence_anchor_guard(row):
    reasons=[];quote=row['quote'];category=row['category']
    if EXCLUDED_MEDIUM.search(row['text']):reasons.append('excluded_or_ambiguous_medium')
    if EXTERNAL_QUOTE.search(quote) and category!='adaptation':reasons.append('explicit_external_or_comparison_quotation')
    clauses=[s for s in re.split(r'[.;!?\n]|,|\b(?:and|but|while|whereas)\s+(?=(?:the|this|its|their|his|her|a|an)\s)',quote,flags=re.I) if s.strip()]
    if category not in ANCHORS or not any(ANCHORS[category].search(c) and (QUALITY.search(c) or (category=='acting' and re.search(JOB,c,re.I))) for c in clauses):
        reasons.append('no_clause_with_both_explicit_aspect_and_quality')
    if category=='emotion' and not EMOTION_EFFECT.search(quote):reasons.append('no_explicit_viewer_or_genre_effect')
    if category=='plot' and re.search(r'\bending\b.{0,35}\b(?:mood|happy|sad|feel)',quote,re.I) and not re.search(r'\b(?:coheren\w*|predictab\w*|contrived|plausib\w*|credib\w*|logic\w*|resol\w*|clich\w*|unresolved|plot\s+holes?)\b',quote,re.I):
        reasons.append('ending_reaction_without_writing_judgment')
    if category=='visuals' and re.search(r'\bsound\s+effects\b',quote,re.I) and not re.search(r'\b(?:visual|special)\s+effects\b',quote,re.I):
        reasons.append('sound_effects_not_visuals')
    return not reasons,reasons
