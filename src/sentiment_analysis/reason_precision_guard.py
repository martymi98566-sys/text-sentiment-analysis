"""Conservative evidence requirements, applied only to existing NLI candidates.

These rules discard ambiguous scope and demand a named aspect plus evaluative
language. They never create labels. Deliberate recall loss is measured separately.
"""
import re

# Ambiguous multi-work reviews are withheld wholesale in this precision mode.
# This can also withhold legitimate adaptation/comparison opinions.
SCOPE_RISK=re.compile(r"\b(?:remake|sequel|prequel|novel|novels|book|books|adaptation|adapted|stage\s+play|stage\s+version|compared?|comparing|unlike|versus|previous\s+(?:film|movie)|earlier\s+(?:film|movie)|original\s+(?:film|movie|version)|(?:first|second|other|another)\s+(?:film|movie|version)|video\s*game|videogame|live\s+concert|concert\s+(?:DVD|video|recording))\b",re.I)
QUALITY=re.compile(r"\b(?:good|bad|great|poor\w*|excellent|wonderful|terrible|brilliant|flat|superb|awful|dull|boring|wooden|coherent|incoherent|original|predictable|unpredictable|clich[eé]\w*|thoughtful|nuanced|shallow|deep|witty|natural|unnatural\w*|stilted|convinc\w*|unconvinc\w*|believ\w*|unbeliev\w*|credible|incredible|amazing|beautiful\w*|ugly|effective|ineffective|weak|strong|outstanding|mediocre|solid|impressive|impressively|lack\w*|fail\w*|skil\w*|faithful\w*|unfaithful\w*|readable|unreadable|unreadably|muddled|repetiti\w*|repeated|rushed|rushing|brisk|drag\w*|engag\w*|annoy\w*|irritat\w*|tedious|slow|sluggish|lousy|horrible|badly|well|best|worst|funny|unfunny|hilarious|scary|frighten\w*|terrify\w*|suspense\w*|thrill\w*|touching|moving|moved|creepy|laugh\w*|no\s+laughs|not\s+funny)\b",re.I)
ANCHORS={
 'plot':r'plot|story|storyline|storytelling|screenplay|script|narrative|ending|premise|twists?',
 'pacing':r'pac(?:e|ed|ing)|editing|tempo|rhythm|drag\w*|rushed|rushing|brisk|repetiti\w*|scenes?\s+(?:are\s+)?repeated|(?:too|overly)\s+(?:slow|long)',
 'acting':r'acting|performance\w*|cast|casting|portray\w*|played|plays|acting\s+role',
 'characters':r'characteri[sz]ation|character\s+development|written\s+(?:motives|motivations|characters)|motivations?|motives|relationships|characters?\s+(?:are|is|were|was)\s+(?:well|badly|poorly)\s+written',
 'dialogue':r'dialogue|dialog|(?:written\s+)?lines|banter|conversations',
 'visuals':r'cinematograph\w*|camera\w*|visual\w*|photograph\w*|lighting|costumes?|makeup|make-up|animation|animated|sets|scenery|special\s+effects|CGI',
 'sound':r'music|soundtrack|score|sound\s+(?:effects|mixing|quality|design)|audio|songs?|recording',
 'emotion':r'funny|unfunny|hilarious|scary|frighten\w*|terrify\w*|suspense\w*|thrill\w*|touching|moving|moved|creepy|laugh\w*|humou?r\w*|jokes',
 'theme':r'themes?|messages?|symbolism|meaning|ideas|social\s+commentary|moral\s+expression|treatment',
 'adaptation':r'adaptation|adapted|faithful|unfaithful',
 'other':r'subtitles?|dubbing|distribution|budget|title|educational\s+value'}
ANCHORS={k:re.compile(r'\b(?:'+v+r')\b',re.I) for k,v in ANCHORS.items()}
QUALITY=re.compile(QUALITY.pattern+r'|\b(?:beautiful\w*|convinc\w*|unconvinc\w*|natural\w*|unnatural\w*|witty|unreadabl\w*|shallow|deep|depth|clever\w*|stilted)\b',re.I)

def conservative_guard(row):
    reasons=[];quote=row['quote'];category=row['category']
    if SCOPE_RISK.search(row['text']):reasons.append('ambiguous_multi_work_or_excluded_medium')
    if category not in ANCHORS or not ANCHORS[category].search(quote):reasons.append('no_explicit_category_anchor')
    if not QUALITY.search(quote):reasons.append('no_explicit_quality_language')
    if category=='visuals' and re.search(r'\bsound\s+effects\b',quote,re.I) and not re.search(r'\b(?:visual|special)\s+effects\b',quote,re.I):
        reasons.append('sound_effects_are_not_visual_effects')
    return not reasons,reasons
