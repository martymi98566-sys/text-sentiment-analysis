"""Frozen small context-NLI correction trial with matched before/after evaluation."""
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_training import load_json, read_jsonl, write_json, digest, metrics
from sentiment_analysis.reason_repair import data, protected_hashes, load_model, score_pairs, fragments_for, aggregate_scores, log
from sentiment_analysis.reason_verified_context import marked_review, hypothesis

CONFIG = ROOT / "configs/reason_context_trial_20261119.json"


def settings():
    config = load_json(CONFIG)
    return config, ROOT / config["output_dir"]


def aspects():
    return {c["id"]: c["aspect"] for c in load_json(ROOT / "configs/review_application.json")["categories"]}


def candidate(review, focus, label, support, kind):
    start = review.index(focus)
    return {"review_text": review, "focus_text": focus, "start": start, "end": start+len(focus),
            "text": marked_review(review, start, start+len(focus)), "label": label,
            "hypothesis": hypothesis(label, aspects()), "expected_support": support, "kind": kind,
            "human_reviewed": False}


def diagnostic_cases():
    rows = []
    def add(review, focus, label, support, kind):
        rows.append(candidate(review, focus or review, label, support, kind))
    for text, label in [("I loved it. Highly recommended.", "plot:positive"),
                        ("I loved it. Highly recommended.", "other:positive"),
                        ("Awful. Avoid this movie.", "plot:negative"),
                        ("Awful. Avoid this movie.", "other:negative")]:
        add(text, None, label, False, "generic")
    for text, label in [("There are many action scenes.", "visuals:positive"),
                        ("The story ends with a wedding.", "plot:positive"),
                        ("The actor walks into the office.", "acting:positive"),
                        ("The soundtrack contains twelve songs.", "sound:positive")]:
        add(text, None, label, False, "factual")
    for review, external, current, pos, neg in [
        ("The first film had wonderful acting. In this sequel, the acting is wooden.",
         "The first film had wonderful acting.", "In this sequel, the acting is wooden.", "acting:positive", "acting:negative"),
        ("The stage play has sharp dialogue. This adaptation's lines are awkward.",
         "The stage play has sharp dialogue.", "This adaptation's lines are awkward.", "dialogue:positive", "dialogue:negative"),
        ("The novel builds complex characters, but this film's characters are flat.",
         "The novel builds complex characters,", "but this film's characters are flat.", "characters:positive", "characters:negative")]:
        add(review, external, pos, False, "external_work")
        add(review, current, neg, True, "current_work_control")
    text = "The story is incoherent, but the soundtrack is excellent."
    add(text, "but the soundtrack is excellent.", "sound:positive", True, "contrast")
    add(text, "but the soundtrack is excellent.", "sound:negative", False, "contrast")
    text = "The acting is terrible, but the music is beautiful."
    add(text, "The acting is terrible,", "acting:negative", True, "contrast")
    add(text, "The acting is terrible,", "sound:positive", False, "outside_focus")
    text = "The lead actor is compelling, but the supporting actors are stiff."
    add(text, None, "acting:positive", True, "both_polarities")
    add(text, None, "acting:negative", True, "both_polarities")
    for text, positive, negative in [("The cinematography is not good.", "visuals:positive", "visuals:negative"),
                                     ("The jokes are not funny.", "emotion:positive", "emotion:negative")]:
        add(text, None, positive, False, "negation")
        add(text, None, negative, True, "negation")
    assert len(rows) == 24
    for i, row in enumerate(rows, 1):
        row["case_id"] = f"CTX-CONTROL-{i:03d}"
    return rows


def supplemental_pairs(reviews):
    definitions = [
        ("AUDIT-029", "the story of Hamlet is definitely one of his best achievements", "plot:positive", "原文在But now on to the movie之前称赞莎士比亚原作，非当前电影。"),
        ("AUDIT-040", "After what I thought was a masterful performance of two roles in Man From Snowy River, WHY was Kirk Douglas replaced by Brian Dennehy in the sequel?", "acting:positive", "masterful评价首部Douglas，不是当前续集Dennehy。"),
        ("AUDIT-059", "I was fascinated with it and thought it was one of the best books I've read recently.", "plot:positive", "明确评价刚读的书，不是之后租的电影。"),
        ("AUDIT-097", "he was so brilliant in 'Kids'", "acting:positive", "明确赞扬另一部作品Kids中的演出，当前Bully演出被批评。"),
        ("AUDIT-076", "If you rent the original film from 1979, though, you will enjoy it, and the actors in it can actually act.", "acting:positive", "原版1979演员好评不属于当前翻拍。"),
        ("AUDIT-067", "the storyline is often too loose and slowly evolving, containing scenes that are unnecessary or just lenghtened too much without any real purpose to the storyline or the character description.", "pacing:negative", "这段描述其他黑泽明电影常见问题，当前Dodesukaden克服了它。"),
        ("AUDIT-001", "The music is good.", "acting:positive", "当前焦点仅配乐；演员好评在全文另一处。"),
        ("AUDIT-001", "The acting by all of these actors is very good.", "sound:positive", "当前焦点仅演员；音乐好评在全文另一处。"),
        ("AUDIT-006", "It has great lines that my family and I quote all the time.", "acting:positive", "明确台词内容，不能搬用另一句acting great。"),
        ("AUDIT-010", "Judd Hirsch and Scott Cohen are great as father and son.", "visuals:positive", "焦点是演员搭档；不能搬用后文布景影像好评。"),
        ("AUDIT-042", "Roll is fast-paced, often funny, and a very worthwhile use of an hour.", "acting:positive", "焦点肯定节奏与幽默，没有演员质量判断。"),
        ("AUDIT-053", "Fantastic production design and suitably bizarre cinematography.", "plot:positive", "焦点是美术摄影；叙事好评位于另一句。"),
        ("AUDIT-095", "The photography was pedestrian.", "acting:positive", "当前句只批评摄影，不能搬用别处Niven好评。"),
        ("AUDIT-080", "The music is also excellent.", "plot:positive", "当前句只肯定音乐，不能搬用全文剧情概念好评。")]
    result = []
    for i, (aid, quote, label, reason) in enumerate(definitions, 1):
        review = reviews[aid]
        row = candidate(review["text"], quote, label, False, "explicit_context_negative")
        result.append({**row, "pair_id": f"CONTEXT-NEG-{i:03d}", "audit_id": aid, "review_id": review["review_id"],
            "target": 1, "verification_basis": reason, "negative_from_label_absence": False,
            "label_source": "assistant_semantic_contrast_review_not_human_gold"})
    return result


def prepare():
    config, output = settings()
    if output.exists():
        raise FileExistsError(output)
    records, names, train, dev, y, audit = data()
    reviewed = ROOT / config["reviewed_data"]
    summary = load_json(reviewed / "preparation_summary.json")
    assert all(digest(ROOT / p) == h for p, h in {**summary["source_sha256"], **summary["original_source_sha256"]}.items())
    pairs = read_jsonl(reviewed / "verified_context_pairs.jsonl")
    assert len(pairs) == 355
    reviews = {r["audit_id"]: r for r in read_jsonl(ROOT / config["audit_dir"] / "reviewed_100_v1.jsonl")}
    extra = supplemental_pairs(reviews)
    pairs += extra
    train_ids = {records[i]["review_id"] for i in train}
    assert {p["review_id"] for p in pairs} <= train_ids
    assert not {p["review_id"] for p in pairs} & {records[i]["review_id"] for i in dev}
    assert all(p["human_reviewed"] is False and p["negative_from_label_absence"] is False for p in pairs)
    empty = np.flatnonzero(y[dev].sum(1) == 0)
    rest = np.flatnonzero(y[dev].sum(1) > 0)
    chosen = sorted(empty.tolist()+np.random.default_rng(20261119).choice(rest, config["fragment_development_subset"]-len(empty), replace=False).tolist())
    assert len(chosen) == config["fragment_development_subset"]
    source_files = [reviewed / "verified_context_pairs.jsonl", reviewed / "preparation_summary.json",
                    ROOT / config["audit_dir"] / "reviewed_100_v1.jsonl", CONFIG,
                    ROOT / config["base_model"] / "model.safetensors"]
    output.mkdir(parents=True)
    (output / "training_pairs.jsonl").write_text("".join(__import__('json').dumps(p, ensure_ascii=False)+"\n" for p in pairs), encoding="utf-8")
    write_json(output / "diagnostic_cases.json", diagnostic_cases())
    write_json(output / "development_subset.json", {"indices_in_development": chosen,
        "review_ids": [records[dev[i]]["review_id"] for i in chosen], "includes_all_no_reason_reviews": int(len(empty)),
        "selection": "All seven no-reason development reviews plus fixed random 57; diagnostic risk-enriched subset, not population estimate."})
    protected = protected_hashes()
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in source_files})
    protocol = {"config": config, "data_audit": audit, "training_pairs": len(pairs),
        "training_reviews": len({p["review_id"] for p in pairs}), "audited_pairs": 355,
        "explicit_supplemental_pairs_from_same_audited_reviews": len(extra),
        "support_pairs": sum(p["target"] == 0 for p in pairs), "not_support_pairs": sum(p["target"] == 1 for p in pairs),
        "protected_sha256": protected, "training_pairs_sha256": digest(output / "training_pairs.jsonl"),
        "diagnostics_sha256": digest(output / "diagnostic_cases.json"), "module_sha256": digest(__file__),
        "development_ids": [records[i]["review_id"] for i in dev], "label_names": names,
        "scope": "Repeatedly used weak-label development agreement; synthetic controls are assistant-authored diagnostics, not independent real-world accuracy.",
        "human_ground_truth": False, "new_real_annotations": 0, "teacher_calls": 0, "serving_model_changed": False}
    write_json(output / "protocol.json", protocol)
    log("context_trial_prepared", pairs=len(pairs), reviews=protocol["training_reviews"], controls=24)


def cap_categories(scores, flags, cap=3):
    flags = flags.copy()
    for i in range(len(flags)):
        present = np.flatnonzero(flags[i].reshape(11, 2).any(1))
        ranking = sorted(present, key=lambda c: (-float(scores[i, 2*c:2*c+2][flags[i, 2*c:2*c+2]].max()), int(c)))
        for c in ranking[cap:]:
            flags[i, 2*c:2*c+2] = False
    return flags


def predictions(scores, threshold, count=None, owners=None, cap=3):
    if owners is not None:
        scores = scores.copy()
        scores[(scores[:, :20] >= threshold).any(1), 20:] = 0
        scores = aggregate_scores(scores, owners, count)
    return cap_categories(scores, scores >= threshold, cap)


def evaluate_metrics(y, scores, threshold, names, owners=None):
    flags = predictions(scores, threshold, len(y), owners)
    result = metrics(y, flags)
    p, r = result["micro_precision"], result["micro_recall"]
    result["micro_f0_5"] = 1.25*p*r/(.25*p+r) if .25*p+r else 0.
    result["threshold"] = threshold
    empty = y.sum(1) == 0
    result["no_reason_reviews"] = int(empty.sum())
    result["no_reason_false_positive_reviews"] = int(flags[empty].any(1).sum())
    result["reviews_with_predictions"] = int(flags.any(1).sum())
    result["mean_predicted_categories"] = float(flags.reshape(-1, 11, 2).any(2).sum(1).mean())
    result["ten_main_categories"] = {k: v for k, v in metrics(y[:, :20], flags[:, :20]).items() if k != "per_label"}
    for name, row in zip(names, result["per_label"]):
        row["label"] = name
    return result


def calibrate(y, scores, names, grid):
    results = [evaluate_metrics(y, scores, t, names) for t in grid]
    selected = max(results, key=lambda r: (r["micro_f0_5"], -abs(r["threshold"]-.5)))
    return {"selected": selected, "at_0_5": next(r for r in results if r["threshold"] == .5),
        "grid": [{k:v for k,v in r.items() if k not in {"per_label", "ten_main_categories"}} for r in results]}


def score_candidates(model, tokenizer, rows, batch_size=24):
    values = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start+batch_size]
            encoded = tokenizer([r["text"] for r in chunk], [r["hypothesis"] for r in chunk], padding=True,
                                truncation=False, return_tensors="pt").to("cuda")
            assert encoded["input_ids"].shape[1] <= 512
            values.append(model(**encoded).logits.float().softmax(-1)[:, 0].cpu().numpy())
    return np.concatenate(values)


def setup():
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")


def evaluate(stage):
    setup()
    config, output = settings()
    destination = output / stage
    destination.mkdir(exist_ok=False)
    records, names, _, dev, y, _ = data()
    protocol = load_json(output / "protocol.json")
    assert protocol["module_sha256"] == digest(__file__)
    assert [records[i]["review_id"] for i in dev] == protocol["development_ids"]
    model_path = ROOT / config["base_model"] if stage == "baseline" else output / "model"
    tokenizer, model = load_model(model_path)
    hs = [hypothesis(n, aspects()) for n in names]
    dev_records = [records[i] for i in dev]
    premises = [marked_review(r["text"], 0, len(r["text"])) for r in dev_records]
    log("context_evaluation_start", stage_name=stage, mode="whole_focus", reviews=len(premises))
    scores = score_pairs(model, tokenizer, premises, hs, config["evaluation_batch_size"])
    np.savez_compressed(destination / "whole_focus_scores.npz", probabilities=scores, targets=y[dev], ids=np.asarray(protocol["development_ids"]))
    result = calibrate(y[dev], scores, names, config["threshold_grid"])
    write_json(destination / "whole_focus_metrics.json", result)
    t = result["selected"]["threshold"]
    if stage != "baseline":
        old_t = load_json(output / "baseline/whole_focus_metrics.json")["selected"]["threshold"]
        write_json(destination / "at_frozen_baseline_threshold.json", evaluate_metrics(y[dev], scores, old_t, names))
    subset = load_json(output / "development_subset.json")["indices_in_development"]
    subset_records = [dev_records[i] for i in subset]
    fragments, owners = fragments_for(subset_records)
    premises = [marked_review(subset_records[owner]["text"], f["start"], f["end"]) for f, owner in zip(fragments, owners)]
    write_json(destination / "fragment_focus_manifest.json", fragments)
    log("context_evaluation_start", stage_name=stage, mode="fragment_focus_64", fragments=len(premises))
    fscores = score_pairs(model, tokenizer, premises, hs, config["evaluation_batch_size"])
    np.savez_compressed(destination / "fragment_focus_scores.npz", probabilities=fscores, targets=y[dev][subset], owners=owners)
    write_json(destination / "fragment_focus_metrics.json", evaluate_metrics(y[dev][subset], fscores, t, names, owners))
    controls = load_json(output / "diagnostic_cases.json")
    probabilities = score_candidates(model, tokenizer, controls)
    results = []
    for row, probability in zip(controls, probabilities):
        results.append({**row, "score": float(probability), "predicted_support": bool(probability >= t),
                        "matches_expected": bool(probability >= t) == row["expected_support"]})
    write_json(destination / "diagnostic_results.json", {"threshold": t, "matched": sum(r["matches_expected"] for r in results),
        "count": len(results), "cases": results, "scope": "Assistant-authored diagnostic expectations, not human-gold accuracy."})
    write_json(destination / "evaluation_complete.json", {"model_sha256": digest(model_path / "model.safetensors"),
        "threshold": t, "no_truncation": True, "whole_development_reviews": 398, "fragment_subset_reviews": 64})
    log("context_evaluation_complete", stage_name=stage, threshold=t, precision=result["selected"]["micro_precision"],
        recall=result["selected"]["micro_recall"], micro_f1=result["selected"]["micro_f1"], diagnostic_matches=sum(r["matches_expected"] for r in results))


def train():
    setup()
    config, output = settings()
    if (output / "model").exists() or (output / "training_history.json").exists():
        raise FileExistsError("Preserve prior training; choose a new output directory")
    assert (output / "baseline/evaluation_complete.json").exists(), "Finish frozen baseline first"
    protocol = load_json(output / "protocol.json")
    assert protocol["config"] == config and protocol["module_sha256"] == digest(__file__)
    assert protocol["training_pairs_sha256"] == digest(output / "training_pairs.jsonl")
    assert protocol["diagnostics_sha256"] == digest(output / "diagnostic_cases.json")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    pairs = read_jsonl(output / "training_pairs.jsonl")
    tokenizer, model = load_model(ROOT / config["base_model"])
    assert model.config.id2label == {0: "entailment", 1: "not_entailment"}
    encoded = tokenizer([p["text"] for p in pairs], [p["hypothesis"] for p in pairs], truncation=False)
    maximum = max(map(len, encoded["input_ids"]))
    assert maximum <= config["max_length"]
    model.get_input_embeddings().weight.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    batch_size, accumulation = config["batch_size"], config["gradient_accumulation"]
    total_steps = math.ceil(math.ceil(len(pairs)/batch_size)/accumulation)*config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*config["warmup_ratio"]), total_steps)
    def batch(indices):
        inputs = tokenizer.pad([{k: v[i] for k, v in encoded.items()} for i in indices], padding=True,
                               pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        target = torch.tensor([pairs[i]["target"] for i in indices], device="cuda")
        return inputs, target
    def training_ce():
        model.eval()
        loss_sum = 0.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(pairs), batch_size):
                ix = list(range(start, min(start+batch_size, len(pairs))))
                inputs, target = batch(ix)
                loss_sum += float(torch.nn.functional.cross_entropy(model(**inputs).logits.float(), target, reduction="sum"))
        return loss_sum/len(pairs)
    started = time.perf_counter()
    initial = training_ce()
    write_json(output / "training_initial.json", {"cross_entropy": initial, "pairs": len(pairs), "maximum_tokens": maximum,
        "optimizer_steps_planned": total_steps, "model_selection": config["checkpoint_selection"]})
    log("context_training_start", pairs=len(pairs), optimizer_steps=total_steps, max_tokens=maximum, initial_train_ce=initial)
    history = []
    for epoch in range(config["epochs"]):
        indices = np.random.default_rng(config["seed"]+epoch).permutation(len(pairs)).tolist()
        chunks = [indices[start:start+batch_size] for start in range(0, len(pairs), batch_size)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, ix in enumerate(chunks):
            inputs, target = batch(ix)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**inputs).logits.float(), target)
            assert torch.isfinite(loss)
            window_start = step//accumulation*accumulation
            denominator = sum(len(c) for c in chunks[window_start:window_start+accumulation])
            (loss*len(ix)/denominator).backward()
            losses.append((float(loss.detach()), len(ix)))
            if (step+1) % accumulation == 0 or step+1 == len(chunks):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1) % 20 == 0:
                log("context_training_progress", epoch=epoch+1, microbatches=step+1, of=len(chunks), elapsed_seconds=round(time.perf_counter()-started, 1))
        row = {"epoch": epoch+1, "training_ce": sum(l*n for l,n in losses)/len(pairs),
               "training_eval_ce": training_ce(), "elapsed_seconds": time.perf_counter()-started}
        history.append(row)
        write_json(output / "training_history.json", history)
        log("context_epoch_end", **row)
    model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")
    write_json(output / "training_complete.json", {"epochs": config["epochs"], "optimizer_steps": total_steps,
        "seconds": time.perf_counter()-started, "model_sha256": digest(output / "model/model.safetensors"),
        "no_development_epoch_selection": True, "source_labels_changed": False, "serving_model_changed": False})
    log("context_training_complete", output=str(output), seconds=round(time.perf_counter()-started, 1))
