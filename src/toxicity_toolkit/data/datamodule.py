"""
DataModule — unified loader interface for different datasets.
"""

from typing import List, Dict
from pathlib import Path
from .adapters.hatexplain import load_hatexplain
from .adapters.jigsaw import load_jigsaw


class DataModule:
    """
    Wraps dataset preparation and access for supported datasets.
    """

    def __init__(self, name: str, out_dir: Path | None = None):
        self.name = name.lower()
        self.out_dir = Path(out_dir or "data") / self.name

    # ------------------------------------------------------------------
    @classmethod
    def from_name(cls, name: str, out_dir: Path | None = None):
        return cls(name, out_dir)

    # ------------------------------------------------------------------
    def prepare(self):
        """
        Download / preprocess dataset and cache locally.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.name == "hatexplain":
            data = load_hatexplain("train")
        elif self.name == "jigsaw":
            data = load_jigsaw("train")
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        # Save a preview so user can verify
        import json
        preview_path = self.out_dir / "preview.jsonl"
        with open(preview_path, "w", encoding="utf-8") as f:
            for row in data[:20]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✅ Saved sample preview to {preview_path}")

    # ------------------------------------------------------------------
    def load(self, split: str = "train") -> List[Dict]:
        """
        Load a split into memory for training or evaluation.
        """
        if self.name == "hatexplain":
            return load_hatexplain(split)
        elif self.name == "jigsaw":
            return load_jigsaw(split)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")
