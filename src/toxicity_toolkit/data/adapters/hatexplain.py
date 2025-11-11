from datasets import load_dataset
def load_hatexplain(split: str = "train"):
    ds = load_dataset("hatexplain", split=split)
    out = []
    for row in ds:
        text = " ".join(row.get("post_tokens", [])) or row.get("text", "")
        labels = ["hate"] if row.get("label", 0) == 1 else []
        out.append({"text": text, "labels": labels})
    return out
