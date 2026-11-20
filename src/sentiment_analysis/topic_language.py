"""Local, text-only language helpers for the exploratory topic-discovery task.

No movie topic names are supplied here.  NLI is used only to distinguish concrete
evaluations from generic opinions/background and to assess fragment polarity.
Its scores are model outputs, not calibrated accuracy or emotional intensity.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from pathlib import Path

import numpy as np


DEFAULT_LANGUAGE_CONFIG = {
    "status_hypotheses": {
        "specific": "The reviewer expresses an opinion about a particular feature or aspect of the movie.",
        "generic": "The reviewer only gives an overall verdict or recommendation.",
        "background": "The reviewer describes the story or background without evaluating it.",
        "uncertain": "The text is unrelated or unintelligible.",
    },
    "polarity_hypotheses": {
        "positive": "This text expresses praise or a positive evaluation of what it describes.",
        "negative": "This text expresses criticism or a negative evaluation of what it describes.",
        "unclear": "This text has no clear positive or negative evaluation, or combines praise with criticism.",
    },
    "status_min_score": 0.50,
    "specific_min_score": 0.50,
    "status_margin": 0.05,
    "polarity_min_score": 0.55,
    "polarity_margin": 0.10,
    "batch_size": 16,
    "max_length": 256,
}


_VERDICT_OBJECT = r"(?:it|this|that|(?:this|that|the) (?:movie|film))"
_VERDICT_ADVERBS = r"(?:(?:really|very|so|quite|simply|absolutely|utterly|truly|pretty|extremely|just|not) )*"
_VERDICT_ADJECTIVE = (
    r"(?:good|great|bad|terrible|awful|excellent|amazing|wonderful|brilliant|fantastic|"
    r"superb|perfect|poor|rubbish|trash|garbage|dreadful|horrible|mediocre|outstanding|disappointing)"
)
_VERDICT_COMPLEMENT = (
    rf"(?:(?:a |an )?{_VERDICT_ADVERBS}{_VERDICT_ADJECTIVE}(?: (?:movie|film))?|"
    r"a masterpiece|a disaster|a waste of time)"
)
_GENERIC_VERDICT_PATTERNS = [re.compile(pattern) for pattern in (
    rf"{_VERDICT_COMPLEMENT}",
    rf"{_VERDICT_OBJECT} (?:is|was|is not|was not) {_VERDICT_COMPLEMENT}",
    rf"(?:what (?:a|an) |(?:such )?(?:a|an|the) ){_VERDICT_ADVERBS}{_VERDICT_ADJECTIVE} (?:movie|film)",
    r"(?:one of )?(?:the )?(?:very )?(?:best|worst) (?:movie|film|movies|films)"
    r"(?: (?:(?:that )?(?:i(?: have| had)?|we(?: have| had)?) (?:ever )?(?:seen|watched)|ever))?",
    rf"(?:i|we) (?:really |absolutely |simply |just )?"
    rf"(?:love|loved|like|liked|hate|hated|dislike|disliked|enjoy|enjoyed) {_VERDICT_OBJECT}",
    rf"(?:(?:i|we) (?:(?:would|can|do|do not|cannot) )?)?"
    rf"(?:highly |strongly |definitely )?recommend(?:ed)?(?: {_VERDICT_OBJECT})?"
    r"(?: to (?:everyone|anyone|all))?",
    rf"(?:please |just )?(?:avoid|skip) {_VERDICT_OBJECT}",
    r"(?:(?:my )?(?:rating|score) ?(?::|is)? ?)?(?:10|[0-9])(?:\.[0-9]+)? ?"
    r"(?:/|out of) ?(?:5|10)(?: (?:stars?|points?))?",
    r"(?:there (?:is|are|was|were)|i (?:saw|found)) "
    r"(?:(?:literally|virtually|absolutely|really) )?no redeeming (?:features|qualities)"
    r"(?: (?:in|about|to) (?:this|that|the) (?:movie|film))?",
)]
_VERDICT_PREFIX = re.compile(
    r"^(?:(?:(?:overall|personally|honestly|well|and|but)|in my opinion|"
    r"i (?:think|feel|believe)(?: that)?)\b[\s,]+)+"
)


def generic_verdict_rule(text):
    """Conservatively recognize complete overall-verdict templates.

    This is a generic sentence-form whitelist, not an aspect dictionary.  All
    clauses must fully match: merely containing 'good', 'worst' or 'recommend'
    never triggers exclusion.  Normalization is used only for matching; callers
    retain the unchanged evidence text.  The rule does not resolve anaphora to
    preceding sentences (e.g. what 'it' refers to).
    """
    if not isinstance(text, str) or not text.strip():
        return False
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'").replace("\u2018", "'")
    for contraction, expansion in (
        ("it's", "it is"), ("that's", "that is"), ("isn't", "is not"),
        ("wasn't", "was not"), ("i've", "i have"), ("we've", "we have"),
        ("i'd", "i would"), ("don't", "do not"), ("can't", "cannot"),
    ):
        normalized = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .!?;,\"'()[]{}")
    normalized = _VERDICT_PREFIX.sub("", normalized)
    if not normalized:
        return False
    # Preserve decimal points in pure ratings such as 8.5/10.
    clauses = re.split(r"(?<!\d)\.|\.(?!\d)|[!?;,]+|\band\b", normalized)
    clauses = [_VERDICT_PREFIX.sub("", value.strip(" .!?;,\"'()[]{}")) for value in clauses]
    clauses = [value for value in clauses if value]
    return bool(clauses) and all(
        any(pattern.fullmatch(value) is not None for pattern in _GENERIC_VERDICT_PATTERNS)
        for value in clauses
    )


def language_config(overrides=None):
    """Return the complete JSON-serializable inference protocol for auditing."""
    config = copy.deepcopy(DEFAULT_LANGUAGE_CONFIG)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown language settings: {sorted(unknown)}")
        for key, value in overrides.items():
            config[key] = copy.deepcopy(value)
    for key, labels in [
        ("status_hypotheses", {"specific", "generic", "background", "uncertain"}),
        ("polarity_hypotheses", {"positive", "negative", "unclear"}),
    ]:
        values = config[key]
        if (not isinstance(values, dict) or set(values) != labels
                or any(not isinstance(value, str) or not value.strip() for value in values.values())):
            raise ValueError(f"Invalid {key}.")
    for key in ["status_min_score", "specific_min_score", "status_margin",
                "polarity_min_score", "polarity_margin"]:
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid threshold: {key}")
    for key in ["batch_size", "max_length"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    return config


def _trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_boundaries(text):
    """Conservative boundaries; all returned spans still refer to source bytes/chars."""
    boundaries = {0, len(text)}
    # HTML breaks are boundaries, but the original markup is not rewritten.
    for match in re.finditer(r"(?:<br\s*/?>)+|\n+", text, flags=re.IGNORECASE):
        boundaries.update([match.start(), match.end()])
    for match in re.finditer(r"[.!?]+[\"')\]]*(?=\s|$)|;(?=\s|$)", text):
        end = match.end()
        before = text[max(0, match.start() - 12):end]
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.$", before, re.IGNORECASE):
            continue
        # A single-letter initial is not a sentence boundary (e.g. J. Smith).
        if re.search(r"\b[A-Z]\.$", before):
            continue
        boundaries.add(end)
    return sorted(boundaries)


def split_review(review_id, text, max_words=90, min_words=1):
    """Split English reviews into source-aligned evidence fragments.

    ``start``/``end`` are Python character offsets in the unchanged input text.
    Contrast conjunctions stay in the following fragment, and negation is never
    stripped.  ``min_words`` is a soft merge preference, never a deletion rule;
    even a one-word review is retained for the information-content classifier.
    """
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")
    if (not isinstance(max_words, int) or not isinstance(min_words, int)
            or min_words <= 0 or max_words < min_words):
        raise ValueError("Invalid fragment word limits.")
    spans = []
    boundaries = _sentence_boundaries(text)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start, end = _trim_span(text, start, end)
        if start == end or re.fullmatch(r"(?:<br\s*/?>)+", text[start:end], re.IGNORECASE):
            continue
        contrast = []
        for match in re.finditer(r"\b(?:but|yet|whereas)\b", text[start:end], re.IGNORECASE):
            position = start + match.start()
            # 'not only X but also Y' is an additive construction, not opposition.
            if re.search(r"\bnot\s+only\b", text[start:position], re.IGNORECASE):
                continue
            if position > start:
                contrast.append(position)
        local_bounds = [start] + contrast + [end]
        for left, right in zip(local_bounds[:-1], local_bounds[1:]):
            left, right = _trim_span(text, left, right)
            if left == right:
                continue
            words = list(re.finditer(r"\S+", text[left:right]))
            if len(words) <= max_words:
                spans.append((left, right))
                continue
            for offset in range(0, len(words), max_words):
                group = words[offset:offset + max_words]
                spans.append((left + group[0].start(), left + group[-1].end()))
    if min_words > 1 and len(spans) > 1:
        merged = []
        for start, end in spans:
            if (merged and len(text[start:end].split()) < min_words
                    and len(text[merged[-1][0]:end].split()) <= max_words):
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        spans = merged
    return [{"fragment_id": f"{review_id}:f{number:04d}", "review_id": str(review_id),
             "start": start, "end": end, "text": text[start:end]}
            for number, (start, end) in enumerate(spans, 1)]


def _validate_score_rows(rows, expected, labels):
    if len(rows) != expected:
        raise ValueError("NLI output changed the input row count.")
    for row in rows:
        if set(row) != set(labels) or any(not np.isfinite(value) or not 0 <= value <= 1
                                          for value in row.values()):
            raise ValueError("Invalid NLI score output.")


def decide_status(scores, config=None):
    config = language_config(config)
    _validate_score_rows([scores], 1, config["status_hypotheses"])
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    best, second = ranked[:2]
    threshold = config["specific_min_score"] if best == "specific" else config["status_min_score"]
    if scores[best] < threshold:
        return "uncertain", "low_model_score"
    if scores[best] - scores[second] < config["status_margin"]:
        return "uncertain", "competing_interpretations"
    return best, "model_decision"


def decide_polarity(scores, config=None):
    config = language_config(config)
    _validate_score_rows([scores], 1, config["polarity_hypotheses"])
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    best, second = ranked[:2]
    if best == "unclear":
        return "unclear", "unclear_or_mixed_evaluation"
    if scores[best] < config["polarity_min_score"]:
        return "unclear", "low_model_score"
    if scores[best] - scores[second] < config["polarity_margin"]:
        return "unclear", "competing_polarities"
    return best, "model_decision"


def classify_fragments(fragments, nli_model, config=None):
    """Add auditable content-status and aspect-fragment polarity decisions.

    Generic opinions/background remain in the returned table.  The caller, not
    this function, decides which records enter topic discovery.  No true review
    label or rating is read or used by this function.
    """
    config = language_config(config)
    result = [dict(fragment) for fragment in fragments]
    if not result:
        return []
    if (any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in result)
            or len({row["fragment_id"] for row in result}) != len(result)):
        raise ValueError("Fragments must have unique IDs and nonempty source text.")
    scores = nli_model.score_hypotheses(
        [row["text"] for row in result], config["status_hypotheses"],
        batch_size=config["batch_size"], max_length=config["max_length"],
    )
    _validate_score_rows(scores, len(result), config["status_hypotheses"])
    for row, values in zip(result, scores):
        row["status"], row["status_reason"] = decide_status(values, config)
        if generic_verdict_rule(row["text"]):
            row["status"] = "generic"
            row["status_reason"] = "generic_template_no_explicit_reason"
        row["status_scores"] = {key: float(value) for key, value in values.items()}
        row["polarity"] = "not_assessed"
        row["polarity_reason"] = "not_a_specific_evaluation"
        row["polarity_scores"] = {}
    selected = [row for row in result if row["status"] == "specific"]
    if selected:
        scores = nli_model.score_hypotheses(
            [row["text"] for row in selected], config["polarity_hypotheses"],
            batch_size=config["batch_size"], max_length=config["max_length"],
        )
        _validate_score_rows(scores, len(selected), config["polarity_hypotheses"])
        for row, values in zip(selected, scores):
            row["polarity"], row["polarity_reason"] = decide_polarity(values, config)
            row["polarity_scores"] = {key: float(value) for key, value in values.items()}
    return result


def _local_path(path):
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {directory}")
    return directory


class LocalEmbeddingModel:
    """Mean-pooled, L2-normalized transformer embeddings; never downloads files."""

    def __init__(self, path, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.path = _local_path(path)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
        ).to(self.device).eval()

    def encode(self, texts, batch_size=32, max_length=256):
        import torch

        texts = list(texts)
        if batch_size <= 0 or max_length <= 0 or any(not isinstance(text, str) for text in texts):
            raise ValueError("Invalid embedding input or batch limits.")
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(texts[start:start + batch_size], padding=True,
                                         truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        result = np.concatenate(vectors).astype(np.float32, copy=False)
        if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-4):
            raise ValueError("Embedding model returned invalid normalized vectors.")
        return result


def entailment_label_ids(id2label):
    """Resolve semantic labels without guessing ambiguous LABEL_0/LABEL_1 IDs."""
    labels = {int(key): re.sub(r"[^a-z]+", "", str(value).lower())
              for key, value in id2label.items()}
    entailment = [key for key, value in labels.items() if value in {"entailment", "entail"}]
    contradiction = [key for key, value in labels.items()
                     if value in {"contradiction", "contradict", "notentailment", "nonentailment"}]
    if len(entailment) != 1:
        raise ValueError(f"NLI model does not identify a unique entailment label: {id2label}")
    return entailment[0], contradiction[0] if len(contradiction) == 1 else None


class LocalNLIModel:
    """Independent hypothesis entailment scores from local NLI weights.

    Default scores retain all NLI classes in the softmax denominator, including
    neutral where present.  They do not sum to one across different hypotheses.
    ``entailment_contradiction`` is an explicit optional HF-style alternative.
    """

    def __init__(self, path, device="cpu", score_mode="entailment"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if score_mode not in {"entailment", "entailment_contradiction"}:
            raise ValueError("Unknown NLI score mode.")
        self.path = _local_path(path)
        self.device = torch.device(device)
        self.score_mode = score_mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
        ).to(self.device).eval()
        self.entailment_id, self.contradiction_id = entailment_label_ids(self.model.config.id2label)
        if score_mode == "entailment_contradiction" and self.contradiction_id is None:
            raise ValueError("The model does not identify a contradiction label.")

    def score_hypotheses(self, texts, hypotheses, batch_size=16, max_length=256):
        import torch

        texts = list(texts)
        if (not isinstance(hypotheses, dict) or not hypotheses or batch_size <= 0 or max_length <= 0
                or any(not isinstance(value, str) or not value.strip() for value in hypotheses.values())
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError("Invalid NLI input, hypotheses, or batch limits.")
        labels = list(hypotheses)
        pairs = [(text, hypotheses[label]) for text in texts for label in labels]
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                encoded = self.tokenizer([pair[0] for pair in chunk], [pair[1] for pair in chunk],
                                         padding=True, truncation="only_first", max_length=max_length,
                                         return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.float()
                if self.score_mode == "entailment_contradiction":
                    probabilities = torch.softmax(logits[:, [self.contradiction_id, self.entailment_id]], dim=1)[:, 1]
                else:
                    probabilities = torch.softmax(logits, dim=1)[:, self.entailment_id]
                scores.extend(float(value) for value in probabilities.cpu().tolist())
        rows = [dict(zip(labels, scores[start:start + len(labels)]))
                for start in range(0, len(scores), len(labels))]
        _validate_score_rows(rows, len(texts), labels)
        return rows
