"""按原项目文本清洗规则匹配 IMDB 原始评分，不做模糊匹配。"""
import hashlib
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass


REVIEW_PATH = re.compile(r"aclImdb/(train|test)/(neg|pos)/(\d+)_(\d+)\.txt")


def clean_review_html(text: str) -> str:
    """保持与 scripts/clean_and_split_data.py 的 clean_html 一致。"""
    text = re.sub(r"</?(?:br|hr|p)\s*/?>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"</?(?:i|em)\s*>", "", text, flags=re.IGNORECASE)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RatingRecord:
    source_path: str
    source_split: str
    label: int
    rating: int
    cleaned_text: str


class RatingIndex:
    """先精确匹配清洗后全文，再只尝试去除全文首尾空白。"""

    def __init__(self, records):
        self.exact = defaultdict(list)
        self.stripped = defaultdict(list)
        self.source_counts = Counter()
        self.record_count = 0
        for record in records:
            if record.source_split not in {"train", "test"}:
                raise ValueError("Unknown original split.")
            if not ((record.label == 0 and 1 <= record.rating <= 4)
                    or (record.label == 1 and 7 <= record.rating <= 10)):
                raise ValueError(f"Rating and polarity disagree: {record.source_path}")
            self.exact[(record.source_split, text_sha256(record.cleaned_text))].append(record)
            self.stripped[(record.source_split, text_sha256(record.cleaned_text.strip()))].append(record)
            self.source_counts[record.source_split] += 1
            self.record_count += 1

    def match(self, text: str, label: int, source_split: str) -> dict:
        if source_split not in {"train", "test"} or label not in {0, 1}:
            raise ValueError("Invalid target split or polarity.")
        candidates = [record for record in self.exact.get((source_split, text_sha256(text)), [])
                      if record.cleaned_text == text]
        mode = "exact_cleaned_text"
        if not candidates:
            candidates = [record for record in self.stripped.get(
                (source_split, text_sha256(text.strip())), [])
                if record.cleaned_text.strip() == text.strip()]
            mode = "outer_whitespace_only" if candidates else "none"
        ratings = sorted({record.rating for record in candidates})
        labels = sorted({record.label for record in candidates})
        if not candidates:
            status = "unmatched"
        elif labels != [label]:
            status = "label_conflict"
        elif len(ratings) != 1:
            status = "rating_conflict"
        else:
            status = "matched_unique" if len(candidates) == 1 else "matched_same_rating"
        accepted = status in {"matched_unique", "matched_same_rating"}
        return {
            "status": status, "match_mode": mode,
            "rating": ratings[0] if accepted else None,
            "candidate_count": len(candidates), "candidate_ratings": ratings,
            "source_paths": sorted(record.source_path for record in candidates),
        }

    def collision_summary(self) -> dict:
        """报告源档案全文重复情况；这些统计不等于目标数据的冲突率。"""
        result = {}
        for name, index in [("exact", self.exact), ("outer_whitespace", self.stripped)]:
            duplicates = [group for group in index.values() if len(group) > 1]
            result[name] = {
                "duplicate_text_groups": len(duplicates),
                "conflicting_rating_groups": sum(len({row.rating for row in group}) > 1
                                                 for group in duplicates),
                "conflicting_label_groups": sum(len({row.label for row in group}) > 1
                                                for group in duplicates),
            }
        return result


def load_rating_index(archive_path) -> RatingIndex:
    """流式读取有标签评论，不解压、不执行档案内容，不使用 unsup 的占位 0 分。"""
    def records():
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                match = REVIEW_PATH.fullmatch(member.name)
                if not member.isfile() or match is None:
                    continue
                if member.size > 1_000_000:
                    raise ValueError(f"Unexpected review size: {member.name}")
                split, polarity, _, rating = match.groups()
                with archive.extractfile(member) as stream:
                    text = stream.read().decode("utf-8")
                yield RatingRecord(member.name, split, int(polarity == "pos"),
                                   int(rating), clean_review_html(text))
    return RatingIndex(records())
