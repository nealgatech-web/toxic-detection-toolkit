from typing import Optional
import json, sys, pathlib
import typer
from rich import print as rprint

from .data.datamodule import DataModule
from .train.trainer import Trainer
from .infer.predictor import Predictor
from .explain.shap_explain import shap_explain_text

app = typer.Typer(help="Toxic Content Detection Toolkit CLI")

@app.command("data")
def data(prepare: bool = typer.Option(False, help="Prepare dataset"),
         dataset: str = typer.Option("hatexplain"),
         out: pathlib.Path = typer.Option(pathlib.Path("data") / "hatexplain")):
    if prepare:
        dm = DataModule.from_name(dataset, out_dir=out)
        dm.prepare()
        rprint(f"[green]Prepared {dataset} at {out}[/green]")

@app.command("train")
def train(dataset: str = "hatexplain",
          model: str = "bert-base-uncased",
          epochs: int = 3,
          output: pathlib.Path = pathlib.Path("runs") / "bert_hx",
          batch_size: int = 16,
          lr: float = 2e-5,
          labels: Optional[str] = None):
    dm = DataModule.from_name(dataset)
    label_list = labels.split(",") if labels else ["hate","harassment","misinformation","spam"]
    trainer = Trainer(model_name=model, labels=label_list, out_dir=output)
    trainer.fit(dm, epochs=epochs, batch_size=batch_size, lr=lr)
    rprint(f"[green]Saved run to {output}[/green]")

@app.command("eval")
def evaluate(run: pathlib.Path, split: str = "validation", threshold: float = 0.5):
    pred = Predictor.from_run(run)
    metrics = pred.evaluate(split=split, threshold=threshold)
    rprint(metrics)

@app.command("infer")
def infer(run: pathlib.Path, threshold: float = 0.5):
    pred = Predictor.from_run(run)
    text = sys.stdin.read().strip()
    result = pred.predict([text], threshold=threshold)[0]
    rprint(json.dumps(result, ensure_ascii=False))

@app.command("explain")
def explain(method: str = "shap", run: pathlib.Path = pathlib.Path("runs/bert_hx"), text: str = "Example"):
    if method.lower() == "shap":
        shap_explain_text(run, text)
    else:
        rprint("[yellow]Only SHAP demo implemented in CLI for now.[/yellow]")

if __name__ == "__main__":
    app()
