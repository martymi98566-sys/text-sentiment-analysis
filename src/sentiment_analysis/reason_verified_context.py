"""Construct partial-label overlays and explicitly verified NLI inputs, never infer negatives."""
from collections import defaultdict

FOCUS_START = "[FOCUS_START]"
FOCUS_END = "[FOCUS_END]"


def marked_review(review, start, end):
    """Preserve every source character once and mark the evaluated passage."""
    if not isinstance(review, str) or not 0 <= start < end <= len(review):
        raise ValueError("Invalid review or focus span")
    if FOCUS_START in review or FOCUS_END in review:
        raise ValueError("Source text collides with focus markers")
    body = review[:start] + FOCUS_START + review[start:end] + FOCUS_END + review[end:]
    return "Review of the current film; only the marked passage is being evaluated:\n" + body


def hypothesis(label, aspects):
    category, sign = label.split(":")
    if sign not in {"positive", "negative"} or category not in aspects:
        raise ValueError("Unknown category/polarity")
    return f"In the marked passage, the reviewer expresses a {sign} opinion about {aspects[category]} of the current film."


def partial_overlay(review):
    """Apply explicit review suggestions in a new overlay; absence remains unknown."""
    combined = defaultdict(list)
    evidence_actions = defaultdict(list)
    excluded = []
    for label in review["label_reviews"]:
        decision, name = label["decision"], label["label"]
        if decision in {"hold", "remove"}:
            excluded.append({"label": name, "decision": decision, "reason": label["reason"]})
            continue
        if decision == "reroute":
            name = label["replacement"]
            quotes = [label["proposed_evidence"]]
        elif decision == "revise_evidence":
            quotes = [label["proposed_evidence"]]
        elif decision == "retain":
            quotes = label["original_evidence"]
        else:
            raise ValueError("Unknown review decision")
        if not name or not quotes or any(not q or q not in review["text"] for q in quotes):
            raise ValueError("Overlay needs source-aligned evidence")
        combined[name].extend(q for q in quotes if q not in combined[name])
        evidence_actions[name].append(decision)
    return {"audit_id": review["audit_id"], "review_id": review["review_id"],
        "labels": [{"category_id": name.split(":")[0], "polarity": name.split(":")[1],
                    "evidence_quotes": quotes, "review_actions": evidence_actions[name]} for name, quotes in combined.items()],
        "unassigned_label_state": "unknown", "complete_reference": False,
        "excluded_original_labels": excluded,
        "pending_addition_candidates": review["addition_candidates"],
        "overall_sentiment_output": False, "human_reviewed": False,
        "provenance": "assistant_review_overlay_not_human_gold", "original_annotation_changed": False}


def build_pair(pair, review, aspects, recovery=None):
    if pair["review_id"] != review["review_id"]:
        raise ValueError("Pair/review mismatch")
    if review["text"][pair["start"]:pair["end"]] != pair["text"]:
        raise ValueError("Focus no longer matches original review")
    target = pair["reviewed_target"]
    if pair["decision"] == "hold":
        if recovery is None:
            return None
        if recovery["support"] not in {"supported", "not_supported"}:
            raise ValueError("Recovery must explicitly adjudicate support")
        if not recovery["context_evidence"] or recovery["context_evidence"] not in review["text"]:
            raise ValueError("Recovery evidence must occur in the full source")
        target = 0 if recovery["support"] == "supported" else 1
    elif recovery is not None:
        raise ValueError("Only a held pair can receive context recovery")
    if target not in {0, 1}:
        raise ValueError("No inferred or unknown training target is allowed")
    premise = marked_review(review["text"], pair["start"], pair["end"])
    return {"pair_id": pair["pair_id"], "review_id": pair["review_id"], "audit_id": pair["audit_id"],
        "text": premise, "hypothesis": hypothesis(pair["label"], aspects), "target": target,
        "label": pair["label"], "focus_text": pair["text"], "focus_start": pair["start"], "focus_end": pair["end"],
        "full_review": review["text"], "original_target": pair["original_target"],
        "audit_decision": pair["decision"], "context_recovered": recovery is not None,
        "verification_basis": recovery["reason"] if recovery else pair["reason"],
        "context_recovery_evidence": recovery["context_evidence"] if recovery else None,
        "human_reviewed": False, "label_source": "assistant_reviewed_weak_supervision",
        "negative_from_label_absence": False}
