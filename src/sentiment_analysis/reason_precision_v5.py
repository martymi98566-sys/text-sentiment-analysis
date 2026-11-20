"""Conservative named-work and emotion boundaries after v4's target-scope failure."""
import re
from sentiment_analysis.reason_precision_v4 import guard as v4_guard
from sentiment_analysis.reason_precision_v4 import verification_candidates as v4_candidates
from sentiment_analysis.reason_precision_v4 import select as v4_select

# Do not guess which named work a review currently evaluates. A selected proof
# containing a quoted proper-name phrase is withheld. This also withholds some
# correct current-title proofs; a different unambiguous proof may still win.
QUOTED_NAME=re.compile(r'''["“«]([^"”»\n]{2,100})["”»]|(?<!\w)'([^'\n]{2,100})'(?!\w)''')
CAPITAL_WORD=re.compile(r'\b[A-Z][a-zA-Z0-9]+\b')
EXPLICIT_EFFECT=re.compile(r'\b(?:funny|funnier|funniest|unfunny|hilarious|amusing|amused|hysterical|scary|scared|frighten\w*|terrify\w*|suspense\w*|thrill\w*|chilling|creepy|heartwarming|heartbreaking|touching|moving|moved|emotionally\s+affecting|humou?r\w*|jokes)\b',re.I)
MOCK_LAUGHTER=re.compile(r'\bcould\s+not\s+even\s+laugh\b|\blaugh\w*\s+at\s+(?:the|this|that|a)\s+(?:sad|bad|terrible|awful|pathetic|ridiculous|poor|sorry)\b',re.I)

def guard(row):
    _,reasons=v4_guard(row);reasons=list(reasons);quote=row['quote']
    for match in QUOTED_NAME.finditer(quote):
        phrase=next(x for x in match.groups() if x is not None)
        words=CAPITAL_WORD.findall(phrase)
        # Single capitals are often scare quotes ("New", "actors"). More than
        # one capitalized word or an all-caps word can name a film, play or book.
        if len(words)>=2 or re.search(r'\b[A-Z]{3,}\b',phrase):
            reasons.append('quoted_named_work_or_ambiguous_proper_name_in_proof');break
    if row['category']=='emotion':
        if not EXPLICIT_EFFECT.search(quote):reasons.append('no_explicit_genre_or_named_emotional_effect')
        if MOCK_LAUGHTER.search(quote):reasons.append('mocking_bad_quality_not_comedy_effect')
    return not reasons,reasons

def verification_candidates(rows,minimum_nli=.8,per_category=2):
    return v4_candidates([r for r in rows if guard(r)[0]],minimum_nli,per_category)

def select(rows,policy):
    return v4_select([r for r in rows if guard(r)[0]],policy)
