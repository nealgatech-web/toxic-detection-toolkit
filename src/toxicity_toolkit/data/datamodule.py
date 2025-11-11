from .adapters.hatexplain import load_hatexplain

class DataModule:
    def __init__(self, name, out_dir=None):
        self.name = name
        self.out_dir = out_dir

    @classmethod
    def from_name(cls, name, out_dir=None):
        return cls(name, out_dir)

    def prepare(self):
        if self.name == "hatexplain":
            _ = load_hatexplain("train")

    def load(self, split="train"):
        if self.name == "hatexplain":
            return load_hatexplain("train" if split=="train" else "validation")
        raise ValueError(f"Unknown dataset {self.name}")
