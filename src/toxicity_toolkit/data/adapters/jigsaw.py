"""
Adapter for Civil Comments (Jigsaw Toxic Comment replacement).
Fully works with Hugging Face `datasets` >= 3.x.
"""

from datasets import load_dataset
from typing import List, Dict

# Map columns to your label set
LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def load_jigsaw(split: str = "train") -> List[Dict]:
    """
    Load and convert Civil Comments dataset into {text, labels[]} schema.
    This is equivalent to the Jigsaw Toxic Comments data.
    """
    ds = load_dataset("civil_comments", split=split)
    out = []

    for row in ds:
        text = row.get("text") or row.get("comment_text") or ""
        labels = []
        for lab in LABELS:
            # the civil_comments labels are probabilities between 0–1
            if float(row.get(lab, 0.0)) >= 0.5:
                labels.append(lab)
        out.append({"text": text, "labels": labels})

    return out
