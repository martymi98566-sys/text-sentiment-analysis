"""Aspect boundary guard used by the serving pipeline."""
import re
from sentiment_analysis.reason_precision_v5 import guard as v5_guard
from sentiment_analysis.reason_precision_v4 import ANCHORS, WRITTEN_DIALOGUE

def type_guard(row):
    _,reasons=v5_guard(row);reasons=list(reasons)
    # Shared subjects and an explicit positive predicate must stay in one unit.
    if row['nli_score']>=.95 and ANCHORS[row['category']].search(row['quote']):
        reasons=[r for r in reasons if r!='no_clause_with_both_explicit_aspect_and_quality']
    if row['category']=='dialogue' and re.search(r'\b(?:hear|audible|inaudible|recording|mixing|volume)\b',row['quote'],re.I) and not WRITTEN_DIALOGUE.search(row['quote']):
        reasons.append('recording_clarity_is_not_written_dialogue_quality')
    return not reasons,reasons
