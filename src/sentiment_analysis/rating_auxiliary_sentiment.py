"""Train-only rating-band supervision with a matched text-only binary control."""
import gc
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from sentiment_analysis.evaluation import classification_metrics
from sentiment_analysis.paths import ROOT
from sentiment_analysis.rating_weighted_sentiment import sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rating_bands(ratings):
    values = np.asarray(ratings, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, [1, 2, 3, 4, 7, 8, 9, 10]).all():
        raise ValueError("Expected ratings 1-4 or 7-10; no neutral ratings.")
    return np.searchsorted([2, 4, 8, 10], values, side="left").astype(np.int64)


def auxiliary_loss(binary_logits, band_logits, labels, bands, coefficient, denominator=None):
    if coefficient < 0 or not math.isfinite(coefficient):
        raise ValueError("Nonnegative finite auxiliary coefficient required.")
    normalizer = len(labels) if denominator is None else denominator
    if normalizer <= 0:
        raise ValueError("Positive group row count required.")
    main = torch.nn.functional.cross_entropy(binary_logits.float(), labels, reduction="sum")
    aux = torch.nn.functional.cross_entropy(band_logits.float(), bands, reduction="sum")
    # Avoid updating unused auxiliary parameters (including optimizer weight decay).
    total = main if coefficient == 0 else main + coefficient * aux
    return total / normalizer, main.detach(), aux.detach()


def selection_key(metrics):
    return (float(metrics["ACCURACY"]), -float(metrics["LOSS"]))


def initial_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RatingAuxiliaryModel(torch.nn.Module):
    """Both heads share exactly the standard DistilBERT classifier features."""

    def __init__(self, sentiment, seed):
        super().__init__()
        self.sentiment = sentiment
        if sentiment.config.model_type != "distilbert":
            raise ValueError("This experiment requires the fixed DistilBERT architecture.")
        # Isolate auxiliary initialization from the main model/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.band_classifier = torch.nn.Linear(sentiment.config.dim, 4)
            generator = torch.Generator().manual_seed(seed + 10000)
            torch.nn.init.normal_(self.band_classifier.weight, std=sentiment.config.initializer_range, generator=generator)
            torch.nn.init.zeros_(self.band_classifier.bias)

    def forward(self, input_ids, attention_mask):
        hidden = self.sentiment.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        pooled = self.sentiment.dropout(torch.nn.functional.relu(self.sentiment.pre_classifier(hidden)))
        return self.sentiment.classifier(pooled), self.band_classifier(pooled)


def load_frame(config, split):
    spec = config["datasets"][split]
    path = ROOT / spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError("Dataset fingerprint changed: " + split)
    frame = pd.read_csv(path)
    if len(frame) != spec["rows"] or frame.text.isna().any() or frame.text.str.strip().eq("").any():
        raise ValueError("Invalid text rows: " + split)
    bands = rating_bands(frame.rating)
    if not np.array_equal((bands >= 2).astype(int), frame.label):
        raise ValueError("Rating/label mismatch.")
    return frame


def encoded_records(frame, tokenizer, max_length):
    features = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length)
    bands = rating_bands(frame.rating)
    return [{"input_ids": ids, "attention_mask": mask, "labels": int(label), "band_labels": int(band)}
            for ids, mask, label, band in zip(features["input_ids"], features["attention_mask"], frame.label, bands)]


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    main_sum, band_sum, band_correct, count = 0., 0., 0, 0
    with torch.inference_mode():
        for batch in loader:
            y = batch.pop("labels").to(device)
            bands = batch.pop("band_labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                binary, auxiliary = model(**inputs)
            binary, auxiliary = binary.float(), auxiliary.float()
            main_sum += float(torch.nn.functional.cross_entropy(binary, y, reduction="sum"))
            band_sum += float(torch.nn.functional.cross_entropy(auxiliary, bands, reduction="sum"))
            band_correct += int((auxiliary.argmax(-1) == bands).sum())
            count += len(y)
            probabilities.extend(binary.softmax(-1)[:, 1].cpu().tolist())
            labels.extend(y.cpu().tolist())
    p = np.asarray(probabilities)
    return {"LOSS": main_sum / count, **classification_metrics(labels, (p > .5).astype(int)),
            "BAND_LOSS": band_sum / count, "BAND_ACCURACY": band_correct / count}, p


def run_name(seed, coefficient):
    return f"seed{seed}_" + ("plain" if coefficient == 0 else f"aux{coefficient:.1f}".replace(".", "p"))


def train(config, seed, coefficient, device="cuda"):
    output = ROOT / config["output_dir"] / run_name(seed, coefficient)
    complete = output / "training_complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text(encoding="utf-8"))
        if previous["config"] != config or previous["seed"] != seed or previous["lambda"] != coefficient:
            raise ValueError("Existing completed run has a different protocol.")
        if previous["checkpoint_sha256"] != sha256(output / "best_model/model.safetensors"):
            raise ValueError("Completed checkpoint changed.")
        print("Reusing verified completed run " + output.name, flush=True)
        return previous
    if output.exists():
        raise FileExistsError("Incomplete run exists; preserve it for inspection: " + str(output))
    output.mkdir(parents=True)
    train_frame = load_frame(config, "train")
    val_frame = load_frame(config, "validation")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["base_model"], local_files_only=True)
    main = AutoModelForSequenceClassification.from_pretrained(
        ROOT / config["base_model"], local_files_only=True, num_labels=2, dtype=torch.float32,
        id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1})
    fingerprint = initial_fingerprint(main)
    model = RatingAuxiliaryModel(main, seed).to(device)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = torch.utils.data.DataLoader(encoded_records(train_frame, tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(encoded_records(val_frame, tokenizer, config["max_length"]),
        batch_size=config["batch_size"], collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    accumulation = config["gradient_accumulation"]
    steps = config["epochs"] * math.ceil(len(loader) / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * steps), steps)
    history, best = [], (-math.inf, -math.inf)
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        main_sum, aux_sum, count = 0., 0., 0
        for step, group in enumerate(itertools.batched(loader, accumulation), 1):
            denominator = sum(len(batch["labels"]) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                y = batch.pop("labels").to(device)
                bands = batch.pop("band_labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    binary, auxiliary = model(**inputs)
                    loss, main_loss, band_loss = auxiliary_loss(binary, auxiliary, y, bands, coefficient, denominator)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss.")
                loss.backward()
                main_sum += float(main_loss)
                aux_sum += float(band_loss)
                count += len(y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step % 200 == 0:
                print(f"{output.name} epoch {epoch}, update {step}/{math.ceil(len(loader) / accumulation)}", flush=True)
        metrics, p = evaluate(model, val_loader, device)
        row = {"RUN": output.name, "SEED": seed, "LAMBDA": coefficient, "EPOCH": epoch,
               "TRAIN_BINARY_LOSS": main_sum / count, "TRAIN_BAND_LOSS": aux_sum / count,
               **{"VALIDATION_" + key: value for key, value in metrics.items()}, "SECONDS": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection_key(metrics) > best:
            best = selection_key(metrics)
            model.sentiment.save_pretrained(output / "best_model")
            tokenizer.save_pretrained(output / "best_model")
            torch.save(model.band_classifier.state_dict(), output / "best_model/auxiliary_head.pt")
            pd.DataFrame({"INPUT ROW": val_frame.input_row, "LABEL": val_frame.label,
                          "POSITIVE PROBABILITY": p, "PREDICTION": (p > .5).astype(int)}).to_csv(output / "validation_predictions.csv", index=False)
            write_json(output / "best_epoch.json", {"epoch": epoch, "validation": metrics})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    result = {"config": config, "seed": seed, "lambda": coefficient, "run": output.name,
              "initial_main_sha256": fingerprint, "checkpoint_sha256": sha256(output / "best_model/model.safetensors"),
              "auxiliary_head_sha256": sha256(output / "best_model/auxiliary_head.pt"),
              "source_sha256": sha256(Path(__file__)), "torch": torch.__version__,
              "features": ["input_ids", "attention_mask"], "test_evaluated_during_training": False,
              "rating_role": "train auxiliary target; never an inference feature"}
    write_json(complete, result)
    del model, main, optimizer, loader, val_loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def pilot(config, device):
    root = ROOT / config["output_dir"]
    seed = config["pilot_seed"]
    train(config, seed, 0., device)
    candidates = []
    for coefficient in config["candidate_lambdas"]:
        meta = train(config, seed, coefficient, device)
        epoch = json.loads((root / meta["run"] / "best_epoch.json").read_text())
        candidates.append({"lambda": coefficient, "run": meta["run"], **epoch})
    winner = max(candidates, key=lambda item: (*selection_key(item["validation"]), -item["lambda"]))
    result = {"config": config, "selected_lambda": winner["lambda"], "candidates": candidates,
              "selected_using": "seed42 binary validation only; no test predictions", "test_seen": False}
    selection = root / "selection.json"
    if selection.exists() and json.loads(selection.read_text()) != result:
        raise ValueError("Refusing to change existing auxiliary selection.")
    write_json(selection, result)
    print("SELECTED AUXILIARY LAMBDA " + str(winner["lambda"]), flush=True)


def confirm(config, device):
    root = ROOT / config["output_dir"]
    selection = json.loads((root / "selection.json").read_text())
    if selection["config"] != config:
        raise ValueError("Selected protocol changed.")
    for seed in config["seeds"]:
        train(config, seed, 0., device)
        train(config, seed, selection["selected_lambda"], device)


def freeze(config):
    root = ROOT / config["output_dir"]
    selection = json.loads((root / "selection.json").read_text())
    runs = []
    for seed in config["seeds"]:
        fingerprints = []
        for coefficient in [0., selection["selected_lambda"]]:
            name = run_name(seed, coefficient)
            path = root / name
            meta = json.loads((path / "training_complete.json").read_text())
            if meta["config"] != config or meta["source_sha256"] != sha256(Path(__file__)):
                raise ValueError("Protocol/source changed during experiment.")
            if meta["checkpoint_sha256"] != sha256(path / "best_model/model.safetensors"):
                raise ValueError("Checkpoint changed.")
            fingerprints.append(meta["initial_main_sha256"])
            runs.append({"run": name, "seed": seed, "lambda": coefficient,
                         "checkpoint_sha256": meta["checkpoint_sha256"],
                         "validation_predictions_sha256": sha256(path / "validation_predictions.csv"),
                         "best_epoch_sha256": sha256(path / "best_epoch.json")})
        if len(set(fingerprints)) != 1:
            raise ValueError("Main model initialization differs within matched seed.")
    result = {"config": config, "selection_sha256": sha256(root / "selection.json"), "runs": runs,
              "frozen_before_test": True, "test_protocol_sha256": sha256(ROOT / config["test_protocol"])}
    path = root / "frozen_models.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Frozen manifest changed.")
    if not path.exists():
        if any(root.glob("*/test_predictions.csv")):
            raise ValueError("Test predictions exist before freezing.")
        write_json(path, result)
    return result


def test(config, device):
    root = ROOT / config["output_dir"]
    manifest = freeze(config)
    protocol = json.loads((ROOT / config["test_protocol"]).read_text(encoding="utf-8"))
    for spec in protocol["sources"].values():
        if sha256(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("Historical comparison source changed.")
    frame = load_frame(config, "test")
    original = pd.read_csv(ROOT / protocol["sources"]["old_test"]["path"]).iloc[frame.input_row.to_numpy()].reset_index(drop=True)
    if not np.array_equal(original.label, frame.label) or not original.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest()).eq(frame.input_text_sha256).all():
        raise ValueError("Common test original text provenance mismatch.")
    frame = frame.copy()
    frame["text"] = original.text.to_numpy()
    for item in manifest["runs"]:
        path = root / item["run"]
        if (path / "test_evaluation.json").exists():
            previous = json.loads((path / "test_evaluation.json").read_text())
            if previous["frozen_manifest_sha256"] != sha256(root / "frozen_models.json") or previous["predictions_sha256"] != sha256(path / "test_predictions.csv"):
                raise ValueError("Existing evaluation changed.")
            print("Reusing verified evaluation " + item["run"], flush=True)
            continue
        if (path / "test_predictions.csv").exists():
            raise FileExistsError("Incomplete evaluation exists.")
        tokenizer = AutoTokenizer.from_pretrained(path / "best_model", local_files_only=True)
        main = AutoModelForSequenceClassification.from_pretrained(path / "best_model", local_files_only=True).to(device)
        main.eval()
        records = tokenizer(frame.text.tolist(), truncation=True, max_length=config["max_length"])
        inputs = [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(records["input_ids"], records["attention_mask"])]
        loader = torch.utils.data.DataLoader(inputs, batch_size=config["batch_size"], collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        probabilities = []
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = main(**{key: value.to(device) for key, value in batch.items()}).logits.float()
                probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        p = np.asarray(probabilities)
        metrics = classification_metrics(frame.label, (p > config["positive_threshold"]).astype(int))
        predictions = pd.DataFrame({"INPUT ROW": frame.input_row, "LABEL": frame.label, "RATING": frame.rating,
                                    "POSITIVE PROBABILITY": p, "PREDICTION": (p > config["positive_threshold"]).astype(int)})
        predictions.to_csv(path / "test_predictions.csv", index=False)
        write_json(path / "test_evaluation.json", {**item, "rows": len(frame), "metrics": metrics,
                   "frozen_manifest_sha256": sha256(root / "frozen_models.json"),
                   "predictions_sha256": sha256(path / "test_predictions.csv"),
                   "rating_used_for_inference": False, "features": ["input_ids", "attention_mask"],
                   "text_policy": protocol["text_policy"]})
        print("TEST " + item["run"] + " " + json.dumps(metrics), flush=True)
        del main, loader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
