"""生成完整带评分数据，显式清除歧义和跨划分重复，保留原始行号。"""
import json
import unicodedata
from collections import defaultdict
from numbers import Integral

import pandas as pd

from sentiment_analysis.ratings import text_sha256


OUTPUT_COLUMNS = [
    "text", "label", "rating", "split", "input_row", "input_text_sha256",
    "text_sha256", "canonical_text_sha256", "source_split", "source_paths",
    "match_mode", "candidate_count", "text_normalized", "encoding_repaired",
]


CP1252_C1_REPAIRS = {}
for codepoint in range(0x80, 0xA0):
    try:
        CP1252_C1_REPAIRS[codepoint] = bytes([codepoint]).decode("cp1252")
    except UnicodeDecodeError:
        # Windows-1252 没有定义的字节保留，由异常字符检查剔除该行。
        pass


def clean_output_text(text: str) -> str:
    """修复可映射的 C1 编码字符、移除零宽空格/BOM，并修剪首尾空白。"""
    return text.translate(CP1252_C1_REPAIRS).replace("\u200b", "").replace("\ufeff", "").strip()


def canonical_text(text: str) -> str:
    """仅用于去重：NFKC、大小写折叠、统一空白，不用于覆盖评论正文。"""
    return " ".join(unicodedata.normalize("NFKC", clean_output_text(text)).casefold().split())


def invalid_text_reason(text) -> str:
    if not isinstance(text, str):
        return "missing_or_nonstring_text"
    cleaned = clean_output_text(text)
    if not cleaned:
        return "blank_text"
    if len(cleaned) > 1_000_000:
        return "oversized_text"
    if "\ufffd" in cleaned:
        return "replacement_character"
    if any(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\t\r\n"
           for char in cleaned):
        return "invalid_control_character"
    if not any(char.isalpha() for char in cleaned):
        return "no_letter_text"
    return ""


def source_conflicts(index) -> dict:
    """跨原始来源检查标准化同文的评分或标签冲突，不能以保留顺序消解。"""
    groups = defaultdict(lambda: {"ratings": set(), "labels": set(), "paths": []})
    for records in index.exact.values():
        for record in records:
            key = text_sha256(canonical_text(record.cleaned_text))
            groups[key]["ratings"].add(record.rating)
            groups[key]["labels"].add(record.label)
            groups[key]["paths"].append(record.source_path)
    return {
        key: {"ratings": sorted(group["ratings"]), "labels": sorted(group["labels"]),
              "paths": sorted(group["paths"])}
        for key, group in groups.items()
        if len(group["ratings"]) > 1 or len(group["labels"]) > 1
    }


def prepare_rating_splits(frames, source_splits, index, priority):
    """保持各组原行序；一致重复按优先级保留一条，任何冲突均不猜测评分。"""
    if len(priority) != len(frames) or set(priority) != set(frames):
        raise ValueError("Split priority must contain every input split exactly once.")
    if set(source_splits) != set(frames):
        raise ValueError("Original source mapping differs from input splits.")
    conflicts = source_conflicts(index)
    rows = []
    for split, frame in frames.items():
        if list(frame.columns) != ["text", "label"]:
            raise ValueError(f"Unexpected input columns: {split}")
        for input_row, row in enumerate(frame.itertuples(index=False)):
            text, label = row.text, row.label
            reason = invalid_text_reason(text)
            item = {
                "split": split, "input_row": input_row, "text": text, "label": label,
                "input_text_sha256": text_sha256(text) if isinstance(text, str) else "",
                "canonical_text_sha256": "", "rating": None, "match_status": "not_attempted",
                "candidate_ratings": [], "source_paths": [], "match_mode": "none",
                "candidate_count": 0, "source_split": source_splits[split],
                "text_normalized": False, "encoding_repaired": False, "drop_reason": reason,
                "kept_split": "", "kept_input_row": None,
            }
            if not reason and (isinstance(label, bool) or not isinstance(label, Integral)
                               or label not in {0, 1}):
                item["drop_reason"] = "invalid_label"
            if not item["drop_reason"]:
                match = index.match(text, int(label), source_splits[split])
                item.update({key: value for key, value in match.items() if key != "status"})
                item["match_status"] = match["status"]
                item["text"] = clean_output_text(text)
                item["text_normalized"] = item["text"] != text
                item["encoding_repaired"] = any(ord(char) in CP1252_C1_REPAIRS for char in text)
                item["canonical_text_sha256"] = text_sha256(canonical_text(item["text"]))
                if match["status"] not in {"matched_unique", "matched_same_rating"}:
                    item["drop_reason"] = match["status"]
                elif item["canonical_text_sha256"] in conflicts:
                    conflict = conflicts[item["canonical_text_sha256"]]
                    item["drop_reason"] = ("source_normalized_label_conflict" if len(conflict["labels"]) > 1
                                           else "source_normalized_rating_conflict")
            rows.append(item)

    groups = defaultdict(list)
    for item in rows:
        if not item["drop_reason"]:
            groups[item["canonical_text_sha256"]].append(item)
    order = {split: position for position, split in enumerate(priority)}
    for group in groups.values():
        if len({(item["label"], item["rating"]) for item in group}) > 1:
            for item in group:
                item["drop_reason"] = "target_normalized_annotation_conflict"
            continue
        keeper = min(group, key=lambda item: (order[item["split"]], item["input_row"]))
        for item in group:
            if item is not keeper:
                item["drop_reason"] = "duplicate_text"
                item["kept_split"] = keeper["split"]
                item["kept_input_row"] = keeper["input_row"]

    outputs = {}
    for split in frames:
        records = []
        for item in rows:
            if item["split"] != split or item["drop_reason"]:
                continue
            records.append({key: item[key] for key in OUTPUT_COLUMNS if key not in {"text_sha256", "source_paths"}}
                           | {"text_sha256": text_sha256(item["text"]),
                              "source_paths": json.dumps(item["source_paths"])})
        outputs[split] = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    audit_columns = ["split", "input_row", "label", "input_text_sha256", "canonical_text_sha256",
                     "rating", "match_status", "match_mode", "candidate_ratings", "source_paths",
                     "drop_reason", "kept_split", "kept_input_row"]
    audit = pd.DataFrame([{key: item[key] for key in audit_columns} for item in rows])
    for name in ["candidate_ratings", "source_paths"]:
        audit[name] = audit[name].apply(json.dumps)
    for name in ["rating", "kept_input_row"]:
        audit[name] = audit[name].astype("Int64")
    return outputs, audit, conflicts


def validate_rating_splits(outputs) -> dict:
    """对内存结果或重新读取的 CSV 执行相同质量约束。"""
    seen_keys, seen_paths = set(), set()
    total = 0
    for split, frame in outputs.items():
        if frame.empty or list(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"Missing rows or unexpected columns: {split}")
        if frame.isna().any().any():
            raise ValueError(f"Missing values in output: {split}")
        if frame["input_row"].duplicated().any() or not frame["input_row"].is_monotonic_increasing:
            raise ValueError(f"Invalid input row references: {split}")
        for row in frame.itertuples(index=False):
            if invalid_text_reason(row.text) or row.text != clean_output_text(row.text):
                raise ValueError(f"Invalid output text: {split}:{row.input_row}")
            if (not isinstance(row.label, Integral) or not isinstance(row.rating, Integral)
                    or not ((row.label == 0 and 1 <= row.rating <= 4)
                            or (row.label == 1 and 7 <= row.rating <= 10))):
                raise ValueError(f"Invalid polarity or rating: {split}:{row.input_row}")
            key = text_sha256(canonical_text(row.text))
            if key in seen_keys:
                raise ValueError("Duplicate normalized text within or across splits.")
            seen_keys.add(key)
            if key != row.canonical_text_sha256 or text_sha256(row.text) != row.text_sha256:
                raise ValueError("Output text fingerprint differs.")
            if row.split != split or row.candidate_count < 1:
                raise ValueError("Invalid output provenance.")
            paths = json.loads(row.source_paths)
            if not paths or len(paths) != row.candidate_count or seen_paths.intersection(paths):
                raise ValueError("Missing or reused original source file.")
            seen_paths.update(paths)
            total += 1
    return {"validated_rows": total, "missing_values": 0, "invalid_text_rows": 0,
            "invalid_label_or_rating_rows": 0, "normalized_duplicate_rows": 0,
            "reused_source_files": 0, "original_row_order_preserved": True}
