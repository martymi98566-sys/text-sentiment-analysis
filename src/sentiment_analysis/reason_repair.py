"""Auditable local diagnosis and NLI repair experiments on a frozen weak-label split."""
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_training import (digest, load_json, metrics, prepare, write_json)
from sentiment_analysis.topic_language import split_review, generic_verdict_rule, entailment_label_ids

BASE = ROOT / "data/reason_training/v2_2000"
OUT = ROOT / "data/reason_training/repair_20261119"
GRID = [round(x / 100, 2) for x in range(5, 100, 5)]
SCOPE = "Same repeatedly used model-annotated development set; agreement, not human accuracy or independent test performance."


def log(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def setup():
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this local protocol")


def data():
    records, excluded, names, train, dev, audit = prepare(load_json(ROOT / "configs/reason_training_v2.json"))
    frozen = load_json(BASE / "split_manifest.json")
    assert [records[i]["review_id"] for i in train] == frozen["train_ids"]
    assert [records[i]["review_id"] for i in dev] == frozen["development_ids"]
    assert names == frozen["label_names"]
    assert audit["source_sha256"] == frozen["source_sha256"]
    y = np.asarray([r["targets"] for r in records], dtype=np.float32)
    saved = np.load(BASE / "best_development_predictions.npz")
    np.testing.assert_array_equal(y[dev], saved["targets"])
    model_config = load_json(BASE / "model/config.json")
    assert [model_config["id2label"][str(i)] for i in range(22)] == names
    rows = [json.loads(s) for s in (BASE / "development_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["review_id"] for r in rows] == frozen["development_ids"]
    for row, i in zip(rows, dev):
        assert row["text"] == records[i]["text"]
        assert set(row["reference_labels"]) == {n for n, f in zip(names, y[i]) if f}
    return records, names, train, dev, y, audit


def protected_hashes():
    paths = list(BASE.rglob("*"))
    paths += list((ROOT / "data/review_application/models/reasons_v1").rglob("*"))
    paths += [ROOT / p for p in ["configs/review_application.json", "src/sentiment_analysis/review_application.py",
                               "src/sentiment_analysis/reason_training.py", "data/pretrained/topic_nli/model.safetensors",
                               "reports/tables/sentiment_comparison_with_rating_auxiliary.csv"]]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def report(y, p, names, threshold, records=None):
    predicted = p >= threshold
    result = metrics(y, predicted)
    result["threshold"] = threshold
    for name, row in zip(names, result["per_label"]):
        row["label"] = name
    if records is not None:
        empty = y.sum(1) == 0
        mixed = (y[:, ::2].sum(1) > 0) & (y[:, 1::2].sum(1) > 0)
        opposite = np.zeros_like(y, dtype=bool)
        for i, row in enumerate(records):
            sign = row["annotation"]["overall_sentiment"]
            if sign == "positive":
                opposite[i, 1::2] = y[i, 1::2] > 0
            elif sign == "negative":
                opposite[i, ::2] = y[i, ::2] > 0
        result["no_reason_reviews"] = int(empty.sum())
        result["no_reason_false_positive_review_rate"] = float(predicted[empty].any(1).mean()) if empty.any() else None
        result["mixed_polarity_reviews"] = int(mixed.sum())
        result["mixed_polarity_micro_f1"] = metrics(y[mixed], predicted[mixed])["micro_f1"] if mixed.any() else None
        result["opposite_to_overall_reference_labels"] = int(opposite.sum())
        result["opposite_to_overall_label_recall"] = float((predicted & opposite).sum() / max(1, opposite.sum()))
        aligned_pred, all_unambiguous = 0, 0
        for row, flags in zip(records, predicted):
            sign = row["annotation"]["overall_sentiment"]
            if sign in {"positive", "negative"}:
                all_unambiguous += int(flags.sum())
                aligned_pred += int(flags[0 if sign == "positive" else 1::2].sum())
        result["predicted_polarity_agrees_with_overall_fraction"] = aligned_pred / max(1, all_unambiguous)
    return result


def calibrate(y, p, names, records=None):
    grid = [report(y, p, names, t, records) for t in GRID]
    chosen = max(grid, key=lambda row: (row["micro_f1"], -abs(row["threshold"] - .5)))
    ranking = []
    for j, name in enumerate(names):
        ranking.append({"label": name, "average_precision": float(average_precision_score(y[:, j], p[:, j])),
                        "roc_auc": float(roc_auc_score(y[:, j], p[:, j])), "prevalence": float(y[:, j].mean())})
    # Deliberately diagnostic only: these 22 fitted thresholds reuse the entire development set.
    per_thresholds = [max(GRID, key=lambda t: (metrics(y[:, j:j+1], p[:, j:j+1] >= t)["micro_f1"], -abs(t-.5))) for j in range(len(names))]
    return {"scope": SCOPE, "selected": chosen, "at_0_5": report(y, p, names, .5, records),
            "threshold_grid": [{k: v for k, v in r.items() if k != "per_label"} for r in grid],
            "ranking": ranking, "macro_average_precision": float(np.mean([r["average_precision"] for r in ranking])),
            "per_label_threshold_diagnostic": {"thresholds": per_thresholds, "optimistic_not_selected_for_serving": True,
                **metrics(y, p >= np.asarray(per_thresholds))}}


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, dtype=torch.float32).to("cuda").eval()
    return tokenizer, model


def score_texts(model, tokenizer, texts, batch_size=16):
    values = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start+batch_size], padding=True, pad_to_multiple_of=8,
                                truncation=False, return_tensors="pt").to("cuda")
            assert encoded["input_ids"].shape[1] <= 384
            values.append(model(**encoded).logits.float().sigmoid().cpu().numpy())
    return np.concatenate(values)


def hypotheses(names):
    categories = load_json(ROOT / "configs/review_application.json")["categories"]
    mapping = {c["id"]: c["aspect"] for c in categories}
    return [f'The reviewer expresses a {name.split(":")[1]} opinion about {mapping[name.split(":")[0]]}.' for name in names]


def score_pairs(model, tokenizer, texts, hs, batch_size=24):
    pairs = [(text, h) for text in texts for h in hs]
    ent, _ = entailment_label_ids(model.config.id2label)
    values, max_len = [], 0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start+batch_size]
            encoded = tokenizer([p[0] for p in chunk], [p[1] for p in chunk], padding=True,
                                truncation=False, return_tensors="pt").to("cuda")
            max_len = max(max_len, encoded["input_ids"].shape[1])
            assert max_len <= 512, "Never silently truncate evidence"
            values.append(model(**encoded).logits.float().softmax(-1)[:, ent].cpu().numpy())
            if start and start % (batch_size*100) == 0:
                log("nli_scoring", pairs_done=start, total_pairs=len(pairs), seconds=round(time.perf_counter()-started, 1))
    log("nli_scoring_complete", pairs=len(pairs), maximum_pair_tokens=max_len, truncated=0)
    return np.concatenate(values).reshape(len(texts), len(hs))


def fragments_for(records):
    fragments, owners = [], []
    for i, row in enumerate(records):
        for f in split_review(row["review_id"], row["text"], max_words=65):
            if not generic_verdict_rule(f["text"]):
                fragments.append(f)
                owners.append(i)
    return fragments, np.asarray(owners, dtype=int)


def aggregate_scores(scores, owners, count):
    result = np.zeros((count, scores.shape[1]), dtype=np.float32)
    np.maximum.at(result, owners, scores)
    return result


def legacy_predictions(scores, owners, count, threshold):
    """Exact serving suppression: other loses only within a fragment with a specific aspect."""
    flags = scores >= threshold
    flags[:, 20:] &= ~flags[:, :20].any(1, keepdims=True)
    return aggregate_scores(flags.astype(np.float32), owners, count) > 0


def probes():
    original = load_json(BASE / "smoke_predictions.json")
    result = [{"review_id": f"probe-{i}", "text": r["text"], "expected": r["assistant_expected_labels"]} for i, r in enumerate(original)]
    extra = [("The acting is excellent, but the plot makes no sense.", ["acting:positive", "plot:negative"]),
             ("The acting is terrible, but the plot makes perfect sense.", ["acting:negative", "plot:positive"]),
             ("The music is beautiful.", ["sound:positive"]), ("The music is terrible.", ["sound:negative"]),
             ("The acting is beautiful.", ["acting:positive"]), ("The acting is terrible.", ["acting:negative"]),
             ("The cinematography is beautiful.", ["visuals:positive"]), ("The cinematography is terrible.", ["visuals:negative"]),
             ("The soundtrack is not good, but the acting is excellent.", ["sound:negative", "acting:positive"]),
             ("The detective walks into a room and talks to a musician.", []),
             ("The movie is terrible. Avoid it.", []),
             ("The acting is not bad, but the dialogue is awful.", ["acting:positive", "dialogue:negative"])]
    for i, (text, expected) in enumerate(extra, len(result)):
        result.append({"review_id": f"probe-{i}", "text": text, "expected": expected})
    return result


def save_probes(path, scores, names, threshold):
    rows = probes()
    for row, values in zip(rows, scores):
        row["predicted"] = [n for n, p in zip(names, values) if p >= threshold]
        row["exact_match"] = set(row["predicted"]) == set(row["expected"])
        row["scores"] = dict(zip(names, map(float, values)))
    write_json(path, {"scope": "Assistant-authored sanity cases, partly previously exposed; not a test set or human gold.",
                      "exact_matches": sum(r["exact_match"] for r in rows), "cases": rows})


def audit():
    OUT.mkdir(parents=True, exist_ok=False)
    records, names, train, dev, y, source_audit = data()
    write_json(OUT / "protected_hashes_before.json", protected_hashes())
    saved = np.load(BASE / "best_development_predictions.npz")
    dev_records = [records[i] for i in dev]
    write_json(OUT / "v2_saved_diagnostics.json", calibrate(y[dev], saved["probabilities"], names, dev_records))
    prevalence = y[train].mean(0)
    constant = np.tile(prevalence, (len(dev), 1))
    sentiment = np.empty_like(constant)
    train_signs = np.asarray([records[i]["annotation"]["overall_sentiment"] for i in train])
    conditional = {s: y[train][train_signs == s].mean(0) for s in set(train_signs)}
    for j, i in enumerate(dev):
        sentiment[j] = conditional.get(records[i]["annotation"]["overall_sentiment"], prevalence)
    write_json(OUT / "prior_diagnostics.json", {"constant_training_priors": calibrate(y[dev], constant, names, dev_records),
        "oracle_overall_sentiment_only": calibrate(y[dev], sentiment, names, dev_records),
        "warning": "Oracle uses development teacher overall sentiment only as a shortcut diagnostic; not a deployable baseline."})
    write_json(OUT / "audit.json", {"scope": SCOPE, "source_audit": source_audit,
        "id_text_label_model_config_and_saved_prediction_alignment": True,
        "training_development_split_exactly_reused": True,
        "development_targets_equal_original_npz": True,
        "human_gold": False, "reserved_evaluation_used": False,
        "training_support_min": int(y[train].sum(0).min()), "training_support_max": int(y[train].sum(0).max()),
        "train_zero_reason": int((y[train].sum(1) == 0).sum()), "dev_zero_reason": int((y[dev].sum(1) == 0).sum()),
        "qualitative_label_noise_example": {"review_id": "RA-TRN-03016-dbc7f344",
            "label": "dialogue:positive", "quote": "This show is a well written drama",
            "assessment": "Writing praise does not explicitly establish dialogue praise; illustrates boundary noise, not a full error-rate estimate."}})
    log("audit_complete", output=str(OUT))


def baseline():
    setup()
    records, names, train, dev, y, _ = data()
    dev_records = [records[i] for i in dev]
    tokenizer, model = load_model(BASE / "model")
    train_scores = score_texts(model, tokenizer, [records[i]["text"] for i in train])
    dev_scores = score_texts(model, tokenizer, [r["text"] for r in dev_records], batch_size=4)
    stored = np.load(BASE / "best_development_predictions.npz")["probabilities"]
    difference = float(abs(stored-dev_scores).max())
    assert difference < 1e-5, difference
    np.savez_compressed(OUT / "v2_training_predictions.npz", probabilities=train_scores, targets=y[train])
    write_json(OUT / "v2_train_development.json", {"train_at_original_threshold": report(y[train], train_scores, names, .45, [records[i] for i in train]),
               "development_at_original_threshold": report(y[dev], dev_scores, names, .45, dev_records), "reloaded_prediction_max_error": difference})
    save_probes(OUT / "v2_probes.json", score_texts(model, tokenizer, [r["text"] for r in probes()], 4), names, .45)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer, model = load_model(ROOT / "data/review_application/models/reasons_v1")
    hs = hypotheses(names)
    full = score_pairs(model, tokenizer, [r["text"] for r in dev_records], hs)
    np.savez_compressed(OUT / "old_nli_full_scores.npz", probabilities=full, targets=y[dev], ids=np.asarray([r["review_id"] for r in dev_records]))
    write_json(OUT / "old_nli_full_metrics.json", calibrate(y[dev], full, names, dev_records))
    fs, owners = fragments_for(dev_records)
    scores = score_pairs(model, tokenizer, [f["text"] for f in fs], hs)
    np.savez_compressed(OUT / "old_nli_fragment_scores.npz", probabilities=scores, owners=owners, targets=y[dev])
    write_json(OUT / "development_fragments.json", fs)
    aggregated = aggregate_scores(scores, owners, len(dev))
    write_json(OUT / "old_nli_fragment_raw_metrics.json", calibrate(y[dev], aggregated, names, dev_records))
    grid = []
    for t in GRID:
        row = metrics(y[dev], legacy_predictions(scores, owners, len(dev), t))
        row["threshold"] = t
        for n, label in zip(names, row["per_label"]):
            label["label"] = n
        grid.append(row)
    chosen = max(grid, key=lambda r: (r["micro_f1"], -abs(r["threshold"]-.5)))
    write_json(OUT / "old_nli_serving_metrics.json", {"scope": SCOPE, "selected": chosen,
        "fixed_application_threshold_0_75": next(r for r in grid if r["threshold"] == .75),
        "threshold_grid": [{k: v for k, v in r.items() if k != "per_label"} for r in grid],
        "inference": "Unchanged serving fragment splitting, generic template exclusion, and per-fragment other suppression. BF16 CUDA scores, max pair length 512 with no truncation."})
    pf, po = fragments_for(probes())
    ps = score_pairs(model, tokenizer, [r["text"] for r in pf], hs)
    aggregated = aggregate_scores(ps, po, len(probes()))
    save_probes(OUT / "old_nli_probes.json", aggregated, names, chosen["threshold"])
    log("baseline_complete", old_serving_micro_f1=chosen["micro_f1"], old_serving_macro_f1=chosen["macro_f1"])


def evidence_contexts(records, names):
    """Use only verbatim training evidence; never infer a missing present label is negative."""
    contexts = []
    for record in records:
        text = record["text"]
        fragments = split_review(record["review_id"], text, max_words=65)
        grouped = {}
        review_labels = {label["category_id"] + ":" + label["polarity"] for label in record["annotation"]["labels"]}
        for label in record["annotation"]["labels"]:
            name = label["category_id"] + ":" + label["polarity"]
            for quote in label["evidence_quotes"]:
                start = text.find(quote)
                assert start >= 0
                end = start + len(quote)
                enclosing = [f for f in fragments if f["start"] <= start and f["end"] >= end]
                if enclosing:
                    span = (enclosing[0]["start"], enclosing[0]["end"])
                else:
                    # Crossing an existing contrast/sentence boundary: retain the complete quote.
                    span = (start, end)
                grouped.setdefault(span, set()).add(name)
        for (start, end), positive in grouped.items():
            contexts.append({"review_id": record["review_id"], "text": text[start:end], "start": start, "end": end,
                             "positive": sorted(positive), "review_labels": sorted(review_labels),
                             "overall_sentiment": record["annotation"]["overall_sentiment"]})
    return contexts


def make_pairs(records, names, cap=64, seed=42):
    rng = random.Random(seed)
    contexts = evidence_contexts(records, names)
    selected = set()
    counts = {}
    for name in names:
        candidates = [i for i, c in enumerate(contexts) if name in c["positive"]]
        # Retain local praise inside negative reviews and local criticism inside positive reviews.
        minority = [i for i in candidates if contexts[i]["overall_sentiment"] in {"positive", "negative"}
                    and contexts[i]["overall_sentiment"] != name.split(":")[1]]
        rng.shuffle(minority)
        chosen = minority[:cap//2]
        remainder = [i for i in candidates if i not in chosen]
        rng.shuffle(remainder)
        chosen += remainder[:max(0, cap-len(chosen))]
        counts[name] = len(chosen)
        selected.update(chosen)
    hs = dict(zip(names, hypotheses(names)))
    pairs = []
    for i in sorted(selected):
        c = contexts[i]
        absent = [name for name in names if name not in c["review_labels"]]
        negative = set()
        for name in c["positive"]:
            opposite = name.split(":")[0] + ":" + ("negative" if name.endswith(":positive") else "positive")
            if opposite in absent:
                negative.add(opposite)
            same_sign = [n for n in absent if n.split(":")[1] == name.split(":")[1]]
            if same_sign:
                negative.add(rng.choice(same_sign))
        for name, target in [(n, 0) for n in c["positive"]] + [(n, 1) for n in sorted(negative)]:
            pairs.append({**c, "label": name, "hypothesis": hs[name], "target": target})
    for r in records:
        if r["annotation"]["information_status"] in {"generic", "background"}:
            for name in rng.sample(names, 4):
                pairs.append({"review_id": r["review_id"], "text": r["text"], "start": 0, "end": len(r["text"]),
                              "positive": [], "review_labels": [], "label": name, "hypothesis": hs[name], "target": 1})
    # Contexts with several evidence spans can yield identical pairs; do not double their weight.
    unique = {(p["review_id"], p["start"], p["end"], p["label"]): p for p in pairs}
    pairs = list(unique.values())
    assert all(p["target"] == 0 or p["label"] not in p["review_labels"] for p in pairs)
    return pairs, {"contexts_available": len(contexts), "contexts_selected": len(selected),
                   "per_label_context_selection_cap": cap, "selected_contexts_per_label_before_union": counts,
                   "pairs": len(pairs), "entailment_pairs": sum(p["target"] == 0 for p in pairs),
                   "non_entailment_pairs": sum(p["target"] == 1 for p in pairs), "training_reviews_with_pairs": len({p["review_id"] for p in pairs})}


def train():
    setup()
    config = load_json(ROOT / "configs/reason_repair_nli_20261119.json")
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    records, names, train_idx, dev_idx, y, audit = data()
    train_records = [records[i] for i in train_idx]
    dev_records = [records[i] for i in dev_idx]
    pairs, pair_audit = make_pairs(train_records, names, config["context_cap_per_label"], config["seed"])
    assert not {r["review_id"] for r in pairs} & {r["review_id"] for r in dev_records}
    text_by_id = {r["review_id"]: r["text"] for r in train_records}
    assert all(text_by_id[p["review_id"]][p["start"]:p["end"]] == p["text"] for p in pairs)
    (output / "training_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in pairs), encoding="utf-8")
    write_json(output / "protocol.json", {"config": config, "pair_audit": pair_audit, "data_audit": audit,
        "module_sha256": digest(__file__), "base_model_sha256": digest(ROOT / config["base_model"] / "model.safetensors"),
        "pair_file_sha256": digest(output / "training_pairs.jsonl"), "development_ids": [r["review_id"] for r in dev_records],
        "selection_evidence": {"new_v2_macro_f1": .2576237582, "old_nli_full_macro_f1": .4452167315,
            "v2_opposite_overall_recall": 4/89, "old_nli_full_opposite_overall_recall": 21/89},
        "scope": SCOPE, "source_labels_changed": False, "new_annotations": False,
        "caveat": "Evidence spans and missing-label negatives inherit teacher boundary/omission errors. Source labels are not repaired or relabeled."})
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer, model = load_model(ROOT / config["base_model"])
    ent, not_ent = entailment_label_ids(model.config.id2label)
    assert (ent, not_ent) == (0, 1)
    encoded = tokenizer([p["text"] for p in pairs], [p["hypothesis"] for p in pairs], truncation=False)
    max_length = max(map(len, encoded["input_ids"]))
    assert max_length <= 512
    model.get_input_embeddings().weight.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    batch_size, accumulation = config["batch_size"], config["gradient_accumulation"]
    steps = math.ceil(math.ceil(len(pairs)/batch_size)/accumulation)*config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*config["warmup_ratio"]), steps)
    started = time.perf_counter()
    history = []
    log("repair_train_start", **pair_audit, max_pair_tokens=max_length, optimizer_steps=steps)
    # A training-only fixed panel verifies task loss actually improves; never used for selection.
    panel = np.random.default_rng(config["seed"]).choice(len(pairs), min(256, len(pairs)), replace=False).tolist()
    def panel_loss():
        model.eval()
        losses = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(panel), batch_size):
                indices = panel[start:start+batch_size]
                batch = tokenizer.pad([{k: v[i] for k, v in encoded.items()} for i in indices],
                                      padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
                target = torch.tensor([pairs[i]["target"] for i in indices], device="cuda")
                losses.append(float(torch.nn.functional.cross_entropy(model(**batch).logits.float(), target, reduction="sum")))
        return sum(losses)/len(panel)
    write_json(output / "training_panel_initial.json", {"indices": panel, "cross_entropy": panel_loss()})
    for epoch in range(config["epochs"]):
        indices = np.random.default_rng(config["seed"]+epoch).permutation(len(pairs)).tolist()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        chunks = [indices[start:start+batch_size] for start in range(0, len(indices), batch_size)]
        for step, ix in enumerate(chunks):
            batch = tokenizer.pad([{k: v[i] for k, v in encoded.items()} for i in ix], padding=True,
                                  pad_to_multiple_of=8, return_tensors="pt").to("cuda")
            target = torch.tensor([pairs[i]["target"] for i in ix], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), target)
            assert torch.isfinite(loss)
            window_start = step//accumulation*accumulation
            window_examples = sum(len(c) for c in chunks[window_start:window_start+accumulation])
            (loss*len(ix)/window_examples).backward()
            losses.append((float(loss.detach()), len(ix)))
            if (step+1) % accumulation == 0 or step+1 == len(chunks):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1) % 100 == 0:
                log("repair_train_progress", epoch=epoch+1, microbatches=step+1, of=len(chunks),
                    seconds=round(time.perf_counter()-started, 1))
        item = {"epoch": epoch+1, "training_ce": sum(l*n for l, n in losses)/len(pairs),
                "training_panel_ce": panel_loss(), "elapsed_seconds": time.perf_counter()-started}
        history.append(item)
        write_json(output / "history.json", history)
        log("repair_epoch_end", **item)
    model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")
    training_seconds = time.perf_counter()-started
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    # Verify a fresh weight reload for all final development predictions.
    tokenizer, model = load_model(output / "model")
    hs = hypotheses(names)
    full = score_pairs(model, tokenizer, [r["text"] for r in dev_records], hs)
    np.savez_compressed(output / "full_scores.npz", probabilities=full, targets=y[dev_idx], ids=np.asarray([r["review_id"] for r in dev_records]))
    write_json(output / "full_metrics.json", calibrate(y[dev_idx], full, names, dev_records))
    fs, owners = fragments_for(dev_records)
    scores = score_pairs(model, tokenizer, [f["text"] for f in fs], hs)
    np.savez_compressed(output / "fragment_scores.npz", probabilities=scores, owners=owners, targets=y[dev_idx])
    aggregated = aggregate_scores(scores, owners, len(dev_idx))
    fragment_result = calibrate(y[dev_idx], aggregated, names, dev_records)
    write_json(output / "fragment_metrics.json", fragment_result)
    # Same legacy suppression protocol as old model, in addition to raw matched comparisons.
    serving_grid = []
    for t in GRID:
        result = report(y[dev_idx], legacy_predictions(scores, owners, len(dev_idx), t).astype(float), names, .5, dev_records)
        result["threshold"] = t
        serving_grid.append(result)
    serving = max(serving_grid, key=lambda r: (r["micro_f1"], -abs(r["threshold"]-.5)))
    write_json(output / "serving_metrics.json", {"selected": serving, "scope": SCOPE,
        "fixed_application_threshold_0_75": next(r for r in serving_grid if r["threshold"] == .75),
        "threshold_grid": [{k: v for k, v in r.items() if k != "per_label"} for r in serving_grid]})
    pf, po = fragments_for(probes())
    ps = score_pairs(model, tokenizer, [r["text"] for r in pf], hs)
    np.savez_compressed(output / "probe_scores.npz", probabilities=ps, owners=po)
    pred = legacy_predictions(ps, po, len(probes()), serving["threshold"]).astype(float)
    save_probes(output / "probes.json", pred, names, .5)
    # Retain raw scores for inspection despite the binary application aggregation above.
    save_probes(output / "raw_fragment_probes.json", aggregate_scores(ps, po, len(probes())), names, fragment_result["selected"]["threshold"])
    write_json(output / "runtime.json", {"training_seconds": training_seconds, "total_seconds": time.perf_counter()-started,
        "model_sha256": digest(output / "model/model.safetensors"), "model_reloaded_for_evaluation": True,
        "no_truncation": True, "application_unchanged": True})
    log("repair_complete", micro_f1=serving["micro_f1"], macro_f1=serving["macro_f1"], output=str(output))
