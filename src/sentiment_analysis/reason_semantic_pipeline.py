"""Local NLI evidence with explicit quality/scope gates and visible long-review windows."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_training import load_json, read_jsonl, write_json, digest
from sentiment_analysis.reason_repair import load_model, score_pairs, hypotheses, log
from sentiment_analysis.reason_context_trial import score_candidates, setup, cap_categories
from sentiment_analysis.reason_verified_context import marked_review, hypothesis
from sentiment_analysis.topic_language import split_review, generic_verdict_rule

QUALITY_HYPOTHESES = [
    "This passage explicitly judges the quality of a particular feature or aspect of the film.",
    "This passage only gives an overall verdict, rating, or recommendation about the film.",
    "This passage describes facts or events without judging the quality of a film aspect."]
SCOPE_HYPOTHESES = [
    "The marked passage evaluates the film currently being reviewed.",
    "The marked passage evaluates a different film, an original book, or a stage play rather than the current film."]


def aspects():
    return {c["id"]: c["aspect"] for c in load_json(ROOT / "configs/review_application.json")["categories"]}


def context_view(text, start, end, tokenizer, longest_hypothesis, limit=512):
    """Keep the whole short review; disclose omitted context for long inputs."""
    full = marked_review(text, start, end)
    def length(premise):
        return len(tokenizer(premise, longest_hypothesis, truncation=False, verbose=False)["input_ids"])
    n = length(full)
    if n <= limit:
        return full, {"windowed": False, "source_ranges": [[0, len(text)]], "source_char_coverage": 1., "pair_tokens": n}
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
    covered = [i for i, (a,b) in enumerate(offsets) if b > start and a < end]
    if not covered:
        raise ValueError("Focus has no source tokens")
    fleft, fright = min(covered), max(covered)+1
    available = 280
    while available >= fright-fleft:
        spare = available-(fright-fleft)
        left = max(0, fleft-spare//2)
        right = min(len(offsets), left+available)
        left = max(0, min(left, right-available))
        a, b = min(start, offsets[left][0]), max(end, offsets[right-1][1])
        anchor_end = offsets[min(35, len(offsets))-1][1]
        if a <= anchor_end:
            a = 0
            prefix, ranges = "", [[0, b]]
        else:
            prefix = "Review opening: "+text[:anchor_end]+"\n[INTERVENING CONTEXT OMITTED]\n"
            ranges = [[0, anchor_end], [a,b]]
        local = text[a:start]+"[FOCUS_START]"+text[start:end]+"[FOCUS_END]"+text[end:b]
        premise = "Review of the current film; only the marked passage is being evaluated:\n"+prefix
        if a:
            premise += "[LOCAL CONTEXT]\n"
        premise += local
        if b < len(text):
            premise += "\n[LATER CONTEXT OMITTED]"
        n = length(premise)
        if n <= limit:
            assert text[start:end] in premise
            return premise, {"windowed": True, "source_ranges": ranges,
                "source_char_coverage": sum(j-i for i,j in ranges)/len(text), "pair_tokens": n}
        available -= 20
    raise ValueError("Focus itself cannot fit; split it before building context")


def safe_fragments(review_id, text, tokenizer):
    result = []
    for original in split_review(review_id, text, max_words=65):
        offsets = tokenizer(original["text"], add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        if len(offsets) <= 150:
            result.append(original)
        else:
            for i in range(0, len(offsets), 150):
                group = offsets[i:i+150]
                a, b = original["start"]+group[0][0], original["start"]+group[-1][1]
                result.append({"review_id": review_id, "fragment_id": original["fragment_id"]+f":t{i//150:02d}", "start": a, "end": b, "text": text[a:b]})
    return [f for f in result if not generic_verdict_rule(f["text"])]


def release(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def score_records(records, destination, screen=.4):
    """Score raw text only; reference labels are never read here."""
    setup()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    names = load_json(ROOT / "data/reason_training/v2_2000/split_manifest.json")["label_names"]
    prior = ROOT / "data/reason_training/repair_20261119/nli_evidence/model"
    current = ROOT / "data/reason_training/context_trial_20261119/model"
    tokenizer, model = load_model(prior)
    hs = hypotheses(names)
    ahs = [hypothesis(n, aspects()) for n in names]
    longest = max(ahs+SCOPE_HYPOTHESES, key=lambda h: len(tokenizer(h)["input_ids"]))
    fragments, owners = [], []
    for i, row in enumerate(records):
        for f in safe_fragments(row["review_id"], row["text"], tokenizer):
            premise, window = context_view(row["text"], f["start"], f["end"], tokenizer, longest)
            assert row["text"][f["start"]:f["end"]] == f["text"]
            fragments.append({**f, "context_text": premise, "context_window": window})
            owners.append(i)
    owners = np.asarray(owners, dtype=np.int32)
    write_json(destination / "fragments.json", fragments)
    started = time.perf_counter()
    log("semantic_local_scoring", reviews=len(records), fragments=len(fragments))
    local = score_pairs(model, tokenizer, [f["text"] for f in fragments], hs, 32)
    selected = np.argwhere(local >= screen)
    promising = sorted(set(int(i) for i,j in selected))
    candidate_rows = [{"text": fragments[int(i)]["context_text"], "hypothesis": ahs[int(j)]} for i,j in selected]
    context_before = np.zeros_like(local)
    context_before[selected[:,0], selected[:,1]] = score_candidates(model, tokenizer, candidate_rows)
    del model
    gc.collect(); torch.cuda.empty_cache()
    log("semantic_current_context_scoring", candidate_pairs=len(candidate_rows))
    tokenizer, model = load_model(current)
    context_after = np.zeros_like(local)
    context_after[selected[:,0], selected[:,1]] = score_candidates(model, tokenizer, candidate_rows)
    del model
    gc.collect(); torch.cuda.empty_cache()
    log("semantic_quality_and_scope_scoring", promising_fragments=len(promising))
    tokenizer, model = load_model(ROOT / "data/pretrained/topic_nli")
    quality = np.zeros((len(fragments), 3), dtype=np.float32)
    scope = np.zeros((len(fragments), 2), dtype=np.float32)
    quality[promising] = score_pairs(model, tokenizer, [fragments[i]["text"] for i in promising], QUALITY_HYPOTHESES, 32)
    scope[promising] = score_pairs(model, tokenizer, [fragments[i]["context_text"] for i in promising], SCOPE_HYPOTHESES, 24)
    np.savez_compressed(destination / "components.npz", local=local, context_before=context_before, context_after=context_after,
                        quality=quality, scope=scope, owners=owners)
    meta = {"reviews": len(records), "fragments": len(fragments), "candidate_pairs": len(candidate_rows),
        "review_ids": [r["review_id"] for r in records], "label_names": names, "candidate_screen": screen,
        "windowed_fragments": sum(f["context_window"]["windowed"] for f in fragments),
        "max_paired_context_tokens": max(f["context_window"]["pair_tokens"] for f in fragments),
        "seconds": time.perf_counter()-started, "silent_truncation": False,
        "module_sha256": digest(__file__), "quality_hypotheses": QUALITY_HYPOTHESES, "scope_hypotheses": SCOPE_HYPOTHESES,
        "model_sha256": {str(p.relative_to(ROOT)):digest(p) for p in [prior/"model.safetensors",current/"model.safetensors",ROOT/"data/pretrained/topic_nli/model.safetensors"]}}
    write_json(destination / "scoring_protocol.json", meta)
    log("semantic_scoring_complete", **{k:meta[k] for k in ["reviews","fragments","candidate_pairs","windowed_fragments","seconds"]})
    return meta


def reference_arrays(rows, names):
    y = np.zeros((len(rows),len(names)), dtype=bool)
    known = np.ones_like(y)
    for i,row in enumerate(rows):
        for r in row["reference_labels"]:
            y[i,names.index(r["label"])] = True
        for name in row["uncertain_labels"]:
            known[i,names.index(name)] = False
    assert not (y & ~known).any()
    return y,known


def infer_components(saved, count, variant, gate, threshold, cap=3):
    local = saved["local"]
    if variant == "local_only":
        scores = local.copy()
    elif variant == "context_before":
        scores = saved["context_before"].copy()
    elif variant == "context_after":
        scores = saved["context_after"].copy()
    elif variant == "dual_before":
        scores = np.minimum(local,saved["context_before"])
    elif variant == "dual_after":
        scores = np.minimum(local,saved["context_after"])
    else:
        raise ValueError(variant)
    flags = (local >= threshold) & (scores >= threshold) if variant.startswith("context") else scores >= threshold
    if gate in {"quality", "quality_scope"}:
        q = saved["quality"]
        accepted = (q[:,0] >= .5) & (q[:,0] >= np.maximum(q[:,1],q[:,2]))
        flags &= accepted[:,None]
    if gate == "quality_scope":
        s = saved["scope"]
        flags &= ((s[:,0] >= .5) & (s[:,0]+.05 >= s[:,1]))[:,None]
    flags[:,20:] &= ~flags[:,:20].any(1,keepdims=True)
    accepted_scores = np.where(flags,scores,0)
    aggregate = np.zeros((count,22),dtype=float)
    np.maximum.at(aggregate,saved["owners"],accepted_scores)
    result = cap_categories(aggregate,aggregate>=threshold,cap)
    return result,aggregate,flags


def masked_metrics(y, pred, known):
    tp=int((y & pred & known).sum()); fp=int((~y & pred & known).sum()); fn=int((y & ~pred & known).sum())
    p=tp/max(1,tp+fp); r=tp/max(1,tp+fn)
    pos_reviews=y.any(1)
    category_y=y.reshape(-1,11,2).any(2)
    category_known=known.reshape(-1,11,2).all(2)|category_y
    category_pred=pred.reshape(-1,11,2).any(2)
    cp=int((category_y&category_pred&category_known).sum()); cf=int((~category_y&category_pred&category_known).sum()); cn=int((category_y&~category_pred&category_known).sum())
    pure_empty=~y.any(1)&known.all(1)
    return {"precision":p,"recall":r,"f1":2*tp/max(1,2*tp+fp+fn),"f0_5":1.25*p*r/(.25*p+r) if .25*p+r else 0,
        "tp":tp,"fp":fp,"fn":fn,"unknown_predictions":int((pred&~known).sum()),
        "positive_review_hit_coverage":float(((y&pred&known).any(1)&pos_reviews).sum()/max(1,pos_reviews.sum())),
        "reviews_with_output":int(pred.any(1).sum()),"mean_categories":float(category_pred.sum(1).mean()),
        "known_category_precision":cp/max(1,cp+cf),"known_category_recall":cp/max(1,cp+cn),
        "definitive_no_reason_reviews":int(pure_empty.sum()),"no_reason_false_positive_reviews":int(pred[pure_empty].any(1).sum())}
