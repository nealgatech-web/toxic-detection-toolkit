from typing import List
import torch, os, json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from ..models.multilabel_bert import MultiLabelBertHead
from ..data.datamodule import DataModule
from .metrics import multilabel_metrics

class SimpleDataset(Dataset):
    def __init__(self, rows, tokenizer, labels):
        self.rows, self.tokenizer, self.labels = rows, tokenizer, labels
        self.lab2id = {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        y = torch.zeros(len(self.labels))
        for l in r.get("labels", []):
            if l in self.lab2id:
                y[self.lab2id[l]] = 1.0
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), y


class Trainer:
    def __init__(self, model_name: str, labels: List[str], out_dir):
        self.labels = labels
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MultiLabelBertHead(model_name, num_labels=len(labels))
        self.model.train()

    def fit(self, dm: DataModule, epochs=3, batch_size=16, lr=2e-5):
        print(f"Loading dataset: {dm.name}")
        rows = dm.load("train")
        ds = SimpleDataset(rows, self.tokenizer, self.labels)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = epochs * len(dl)
        sched = get_linear_schedule_with_warmup(opt, 0, total_steps)
        bce = torch.nn.BCEWithLogitsLoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        for ep in range(epochs):
            total_loss = 0.0
            for input_ids, attn, y in dl:
                input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
                logits = self.model(input_ids, attn)
                loss = bce(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {ep+1}/{epochs} - Loss: {total_loss/len(dl):.4f}")

        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(self.out_dir)
        with open(os.path.join(self.out_dir, "labels.json"), "w") as f:
            json.dump(self.labels, f)
        print(f"✅ Model saved to {self.out_dir}")
