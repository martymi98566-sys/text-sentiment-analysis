"""按固定协议补齐全部评分，输出独立清洗数据和逐行审计。"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from sentiment_analysis.paths import CONFIGS_DIR, ROOT
from sentiment_analysis.rating_preparation import prepare_rating_splits, validate_rating_splits
from sentiment_analysis.ratings import load_rating_index


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_dataset(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = {"archive": config["archive"], **config["splits"]}
    for name, record in records.items():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Fixed input changed: {name}")
    if config["cleaning_policy_version"] != 1:
        raise ValueError("Unsupported cleaning policy.")
    index = load_rating_index(ROOT / config["archive"]["path"])
    if dict(index.source_counts) != {"train": 25000, "test": 25000}:
        raise ValueError("Unexpected archive counts.")
    frames = {name: pd.read_csv(ROOT / record["path"]) for name, record in config["splits"].items()}
    original_sources = {name: record["source_split"] for name, record in config["splits"].items()}
    outputs, audit, conflicts = prepare_rating_splits(frames, original_sources, index,
                                                   config["duplicate_keep_priority"])
    quality = validate_rating_splits(outputs)
    summary_rows = []
    for name, frame in outputs.items():
        original_count = len(frames[name])
        retention = len(frame) / original_count
        if retention < config["minimum_retention_per_split"]:
            raise ValueError(f"Too many rows removed from {name}: retention={retention:.4%}")
        summary_rows.append({
            "SPLIT": name, "INPUT ROWS": original_count, "OUTPUT ROWS": len(frame),
            "REMOVED": original_count - len(frame), "RETENTION": retention,
            "NEGATIVE": int((frame["label"] == 0).sum()), "POSITIVE": int((frame["label"] == 1).sum()),
            "NORMALIZED TEXT": int(frame["text_normalized"].sum()),
            "ENCODING REPAIRED": int(frame["encoding_repaired"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    dropped = audit.loc[audit["drop_reason"] != ""].copy()
    reasons = (dropped.groupby(["split", "drop_reason"]).size().rename("COUNT")
               .reset_index().rename(columns={"split": "SPLIT", "drop_reason": "REASON"}))
    distributions = pd.concat([frame.groupby(["label", "rating"]).size().rename("COUNT").reset_index()
                               .assign(SPLIT=name) for name, frame in outputs.items()], ignore_index=True)
    source_risks = pd.DataFrame([
        {"CANONICAL TEXT SHA256": key, "CANDIDATE RATINGS": json.dumps(value["ratings"]),
         "CANDIDATE LABELS": json.dumps(value["labels"]), "SOURCE PATHS": json.dumps(value["paths"])}
        for key, value in sorted(conflicts.items())
    ], columns=["CANONICAL TEXT SHA256", "CANDIDATE RATINGS", "CANDIDATE LABELS", "SOURCE PATHS"])
    paths = config["outputs"]
    targets = {name: ROOT / path for name, path in paths["datasets"].items()}
    protected = {(ROOT / record["path"]).resolve() for record in records.values()}
    output_frames = {**{targets[name]: frame for name, frame in outputs.items()},
                     ROOT / paths["audit"]: audit, ROOT / paths["dropped"]: dropped,
                     ROOT / paths["summary"]: summary, ROOT / paths["reasons"]: reasons,
                     ROOT / paths["rating_distribution"]: distributions,
                     ROOT / paths["source_conflicts"]: source_risks}
    output_paths = [path.resolve() for path in output_frames] + [(ROOT / paths["metadata"]).resolve()]
    if len(set(output_paths)) != len(output_paths) or protected.intersection(output_paths):
        raise ValueError("Output paths overlap one another or protected input files.")
    output_records = {}
    for path, frame in output_frames.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        output_records[path.relative_to(ROOT).as_posix()] = {"sha256": sha256(path), "rows": len(frame)}
    reloaded = {name: pd.read_csv(path) for name, path in targets.items()}
    if validate_rating_splits(reloaded) != quality:
        raise ValueError("CSV round-trip quality checks differ.")
    for record in records.values():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("An original input changed.")
    code_paths = [Path(__file__).resolve(), ROOT / "src/sentiment_analysis/rating_preparation.py",
                  ROOT / "src/sentiment_analysis/ratings.py", ROOT / "scripts/clean_and_split_data.py"]
    metadata = {
        "config_sha256": sha256(config_path), "inputs": records, "outputs": output_records,
        "quality_checks": quality, "source_normalized_conflict_groups": len(conflicts),
        "input_rows": len(audit), "retained_rows": sum(len(frame) for frame in outputs.values()),
        "removed_rows": len(dropped), "duplicate_keep_priority": config["duplicate_keep_priority"],
        "source_review_counts": dict(index.source_counts),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas"]},
        "code_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths},
        "training_performed": False, "original_inputs_overwritten": False,
        "semantic_annotation_correctness_manually_verified": False,
    }
    source_record = ROOT / "data/local_setup/rating_archive_source.json"
    if source_record.exists():
        metadata["archive_download"] = json.loads(source_record.read_text(encoding="utf-8"))
    metadata_path = ROOT / paths["metadata"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("Removal reasons:")
    print(reasons.to_string(index=False))
    print(json.dumps(quality, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "rating_dataset_preparation.json")
    args = parser.parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()
