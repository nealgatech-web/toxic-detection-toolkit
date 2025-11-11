import os, json, torch
import numpy as np
from transformers import AutoTokenizer
from ..models.multilabel_bert import MultiLabelBertHead

class Predictor:
    def __init__(self, tokenizer, model, labels):
        self.tokenizer = tokenizer
        self.model = model
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)

    @classmethod
    def from_run(cls, run_dir):
        run_dir = str(run_dir)
        labels_path = os.path.join(run_dir, "labels.json")
        model_path = os.path.join(run_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model checkpoint at {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels.json at {labels_path}")

        with open(labels_path, "r") as f:
            labels = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(run_dir)
        model = MultiLabelBertHead("bert-base-uncased", num_labels=len(labels))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return cls(tokenizer, model, labels)

    def predict(self, texts, threshold=0.5):
        """Predict labels for a list of input texts."""
        results = []
        for text in texts:
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
            labels = [self.labels[i] for i, p in enumerate(probs) if p >= threshold]
            results.append(
                {
                    "text": text,
                    "scores": dict(zip(self.labels, probs.tolist())),
                    "labels": labels,
                }
            )
        return results

    def evaluate(self, split="validation", threshold=0.5):
        """Placeholder evaluation for CLI. Extend with your validation logic."""
        print(f"Evaluating on split: {split} (threshold={threshold})")
        return {"split": split, "threshold": threshold, "macro_f1": 0.0}
