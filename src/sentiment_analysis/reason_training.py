"""Train a text-only 22-label reason classifier on explicitly sourced weak labels."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sentiment_analysis.paths import ROOT


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(y, predictions):
    y, predictions = np.asarray(y, dtype=bool), np.asarray(predictions, dtype=bool)
    tp = (y & predictions).sum(axis=0)
    fp = (~y & predictions).sum(axis=0)
    fn = (y & ~predictions).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    return {"micro_precision": float(tp.sum() / max(1, (tp + fp).sum())),
            "micro_recall": float(tp.sum() / max(1, (tp + fn).sum())),
            "micro_f1": float(2 * tp.sum() / max(1, (2 * tp + fp + fn).sum())),
            "macro_f1": float(f1.mean()), "exact_match": float((y == predictions).all(axis=1).mean()),
            "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
            "mean_reference_labels": float(y.sum(axis=1).mean()),
            "per_label": [{"precision": float(p), "recall": float(r), "f1": float(f),
                           "support": int(t + n), "true_positive": int(t), "false_positive": int(b), "false_negative": int(n)}
                          for p, r, f, t, b, n in zip(precision, recall, f1, tp, fp, fn)]}


def encode_labels(annotation, label_names):
    present = {item["category_id"] + ":" + item["polarity"] for item in annotation["labels"]}
    if not present <= set(label_names):
        raise ValueError("Unknown category/polarity in training source.")
    return [float(name in present) for name in label_names]


def usable(annotation):
    return (annotation["review_status"] == "completed" and annotation["information_status"] != "uncertain"
            and annotation["overall_sentiment"] != "uncertain" and not annotation["uncertainty_notes"]
            and not annotation.get("uncertainty_fragments"))


def split_indices(y, fraction, seed, candidates=32):
    """Choose a fixed text-disjoint split by label balance only, before any training."""
    y = np.asarray(y)
    total = y.sum(axis=0)
    best = None
    for offset in range(candidates):
        train, development = train_test_split(np.arange(len(y)), test_size=fraction, random_state=seed + offset)
        support = y[development].sum(axis=0)
        train_support = y[train].sum(axis=0)
        missing = int(((support == 0) & (total >= 2)).sum() + ((train_support == 0) & (total >= 2)).sum())
        imbalance = float(np.square((support - total * len(development) / len(y)) / np.sqrt(np.maximum(total, 1))).mean())
        key = (missing, imbalance, offset)
        if best is None or key < best[0]:
            best = (key, train, development)
    return best[1], best[2], {"seed": seed + best[0][2], "candidate_count": candidates,
                             "missing_label_cells": best[0][0], "label_balance_error": best[0][1],
                             "selection_used_model_metrics": False}


def prepare(config):
    # Reuse the already-tested mechanical gates, without running their output-writing CLI.
    from check_reason_annotation_batches import scan
    from validate_external_reason_review import validate
    source = ROOT / config["annotation_dir"]
    package = source / "quality_repair_42"
    checks, progress, _ = scan(package)
    if len(checks) != 42 or not all(row["complete"] for row in checks):
        raise ValueError("Repair annotations have not passed all 42 batches.")
    inputs = load_json(source / "all_2000_inputs.json")
    annotations = read_jsonl(source / "existing_320_model_reviews.jsonl")
    paths = [source / "all_2000_inputs.json", source / "existing_320_model_reviews.jsonl"]
    for number in range(1, 43):
        path = package / f"results/annotations_batch_{number:03d}.jsonl"
        annotations.extend(read_jsonl(path))
        paths.append(path)
    categories = load_json(ROOT / config["application_config"])["categories"]
    validate({"batch_id": "reason_v2_all2000", "categories": categories,
              "require_uncertainty_fragments": True, "reviews": inputs}, annotations)
    if len(inputs) != 2000 or [r["review_id"] for r in inputs] != [r["review_id"] for r in annotations]:
        raise ValueError("All 2000 inputs and annotations must align exactly.")
    names = [category["id"] + ":" + sign for category in categories for sign in ["positive", "negative"]]
    reserved = []
    reservation_dir = source.parent
    reservation_manifest = load_json(reservation_dir / "reservation_manifest.json")
    for split in ["validation", "test"]:
        path = reservation_dir / (split + "_reservation.jsonl")
        if digest(path) != reservation_manifest[split + "_sha256"]:
            raise ValueError("Original evaluation reservation changed.")
        reserved.extend(read_jsonl(path))
        paths.append(path)
    from prepare_deepseek_desktop_handoff import canonical
    normalized = [canonical(row["text"]) for row in inputs]
    if len(set(normalized)) != 2000 or set(normalized) & {canonical(row["text"]) for row in reserved}:
        raise ValueError("Duplicate training text or held-out reservation overlap.")
    accepted, excluded = [], []
    for item, annotation in zip(inputs, annotations):
        if usable(annotation):
            accepted.append(dict(item, targets=encode_labels(annotation, names), annotation=annotation))
        else:
            excluded.append({"review_id": item["review_id"], "reason": "unresolved_or_out_of_domain",
                             "notes": annotation["uncertainty_notes"]})
    y = np.asarray([row["targets"] for row in accepted], dtype=np.float32)
    train, development, split_audit = split_indices(y, config["development_fraction"], config["seed"], config["split_candidates"])
    if split_audit["missing_label_cells"]:
        raise ValueError("A label is missing from one split; revise split protocol before training.")
    return accepted, excluded, names, train, development, {"source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
             "usable_reviews": len(accepted), "excluded_reviews": len(excluded), "training_reviews": len(train),
             "development_reviews": len(development), "split": split_audit, "reserved_evaluation_reviews": len(reserved),
             "reserved_evaluation_scored": False, "human_ground_truth": False, "scope": config["evaluation_scope"]}


def run(config, resume=False):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("This training protocol requires the available CUDA GPU.")
    output = ROOT / config["output_dir"]
    if output.exists() and not resume:
        raise FileExistsError("Preserve prior run; use --resume for its saved epoch state.")
    records, excluded, names, train_idx, dev_idx, audit = prepare(config)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["base_model"], local_files_only=True)
    encoded = tokenizer([row["text"] for row in records], truncation=False, return_length=True, verbose=False)
    lengths = encoded.pop("length")
    if max(lengths) > config["max_length"]:
        raise ValueError("Maximum length would discard annotated context; revise protocol instead of silently truncating.")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        previous = load_json(output / "protocol.json")
        if previous["config"] != config or previous["audit"]["source_sha256"] != audit["source_sha256"]:
            raise ValueError("Resume data/config no longer matches the frozen run.")
    else:
        write_json(output / "protocol.json", {"config": config, "audit": audit,
                   "architecture": "DeBERTa-v3-base encoder + 22 independent sigmoid outputs; weighted BCE training.",
                   "classification_head": "New randomly initialized 22-label head; old two-class NLI head not reused.",
                   "max_observed_token_length": max(lengths), "truncated_reviews": 0,
                   "source_code_sha256": {"module": digest(__file__)}})
        write_json(output / "excluded_reviews.json", excluded)
        write_json(output / "split_manifest.json", {"train_ids": [records[i]["review_id"] for i in train_idx],
                   "development_ids": [records[i]["review_id"] for i in dev_idx], "label_names": names, **audit})
    y = np.asarray([row["targets"] for row in records], dtype=np.float32)
    train_support = y[train_idx].sum(axis=0)
    pos_weight = np.clip(np.sqrt((len(train_idx) - train_support) / np.maximum(train_support, 1)), 1, config["positive_weight_cap"])
    write_json(output / "label_distribution.json", [{"label": name, "train_positive": int(a), "development_positive": int(b), "positive_weight": float(w)}
               for name, a, b, w in zip(names, train_support, y[dev_idx].sum(axis=0), pos_weight)])
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / config["base_model"], local_files_only=True,
            num_labels=len(names), id2label={i: name for i, name in enumerate(names)}, label2id={name: i for i, name in enumerate(names)},
            problem_type="multi_label_classification", ignore_mismatched_sizes=True, dtype=torch.float32).to("cuda")
    if config["freeze_word_embeddings"]:
        model.get_input_embeddings().weight.requires_grad_(False)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    def collate(indices):
        batch = tokenizer.pad([{key: values[int(i)] for key, values in encoded.items()} for i in indices],
                              padding=True, pad_to_multiple_of=8, return_tensors="pt")
        return batch, torch.tensor(y[np.asarray(indices)], dtype=torch.float32)
    dev_loader = DataLoader(dev_idx, batch_size=config["batch_size"], collate_fn=collate)
    microbatches = math.ceil(len(train_idx) / config["batch_size"])
    total_steps = math.ceil(microbatches / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    weighted_bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device="cuda"))
    def evaluate():
        model.eval()
        logits = []
        with torch.inference_mode():
            for batch, _ in dev_loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits.append(model(**batch.to("cuda")).logits.float().cpu())
        logits = torch.cat(logits)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite development logits.")
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y[dev_idx])))
        return loss, logits.sigmoid().numpy()
    start_epoch, best_loss, history = 0, float("inf"), []
    if resume:
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_loss, history = state["epoch"], state["best_loss"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    else:
        loss, probabilities = evaluate()
        write_json(output / "initial_development_metrics.json", {"loss": loss, **metrics(y[dev_idx], probabilities >= .5), "scope": config["evaluation_scope"]})
    started = time.perf_counter()
    print(json.dumps({"stage": "training_start", **{k: audit[k] for k in ["usable_reviews", "excluded_reviews", "training_reviews", "development_reviews"]},
                      "gpu": torch.cuda.get_device_name(0), "token_max": max(lengths), "labels": len(names)}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.perf_counter()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(train_idx, batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collate)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (batch, targets) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch.to("cuda")).logits
                loss = weighted_bce(logits.float(), targets.to("cuda"))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; model not accepted.")
            window_start = ((step - 1) // config["gradient_accumulation"]) * config["gradient_accumulation"]
            window_examples = min(config["gradient_accumulation"] * config["batch_size"],
                                  len(train_idx) - window_start * config["batch_size"])
            (loss * len(targets) / window_examples).backward()
            losses.append(float(loss.detach()))
            if step % config["gradient_accumulation"] == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 40 == 0 or step == len(loader):
                print(json.dumps({"stage": "progress", "epoch": epoch + 1, "microbatch": step, "of": len(loader),
                                  "mean_loss": sum(losses)/len(losses), "seconds": round(time.perf_counter()-epoch_start, 1)}, ensure_ascii=False), flush=True)
        dev_loss, probabilities = evaluate()
        score = metrics(y[dev_idx], probabilities >= .5)
        item = {"epoch": epoch + 1, "training_loss": sum(losses)/len(losses), "development_bce": dev_loss,
                **{k: v for k, v in score.items() if k != "per_label"}, "seconds": time.perf_counter()-epoch_start}
        history.append(item)
        if dev_loss < best_loss:
            best_loss = dev_loss
            model.save_pretrained(output / "model")
            tokenizer.save_pretrained(output / "model")
            np.savez_compressed(output / "best_development_predictions.npz", probabilities=probabilities, targets=y[dev_idx])
            write_json(output / "best_epoch.json", item)
        write_json(output / "history.json", history)
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_loss": best_loss, "history": history,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, output / "resume.pt")
        print(json.dumps({"stage": "epoch_end", **item}, ensure_ascii=False), flush=True)
    prediction_file = np.load(output / "best_development_predictions.npz")
    probabilities = prediction_file["probabilities"]
    grid = [{"threshold": value, **{k: v for k, v in metrics(y[dev_idx], probabilities >= value).items() if k != "per_label"}}
            for value in config["threshold_grid"]]
    chosen = max(grid, key=lambda row: (row["micro_f1"], -abs(row["threshold"]-.5)))
    final = metrics(y[dev_idx], probabilities >= chosen["threshold"])
    for name, row in zip(names, final["per_label"]):
        row["label"] = name
    write_json(output / "threshold_search.json", grid)
    write_json(output / "inference_config.json", {"label_names": names, "threshold": chosen["threshold"],
               "max_length": config["max_length"], "model_path": str((output/"model").relative_to(ROOT)), "scope": config["evaluation_scope"]})
    result = {"status": "completed", **audit, "metrics": final, "threshold": chosen["threshold"],
              "threshold_0_5_metrics": metrics(y[dev_idx], probabilities >= .5),
              "selected_epoch": load_json(output / "best_epoch.json")["epoch"], "history": history,
              "seconds_this_invocation": time.perf_counter()-started, "model_saved": True,
              "model_sha256": digest(output / "model/model.safetensors"),
              "serving_model_changed": False, "ratings_used": False, "external_test_used": False}
    write_json(output / "metrics.json", result)
    output_rows = []
    for index, scores in zip(dev_idx, probabilities):
        record = records[index]
        output_rows.append({"review_id": record["review_id"], "text": record["text"],
                            "reference_labels": [name for name, flag in zip(names, record["targets"]) if flag],
                            "predicted_labels": [name for name, value in zip(names, scores) if value >= chosen["threshold"]],
                            "scores": {name: float(value) for name, value in zip(names, scores)}})
    (output / "development_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in output_rows), encoding="utf-8")
    print(json.dumps({"stage": "completed", "output": str(output), "threshold": chosen["threshold"],
                      **{k: v for k, v in final.items() if k != "per_label"}}, ensure_ascii=False), flush=True)
    return result
