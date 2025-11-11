from typing import Optional
import json, sys, pathlib
import typer
from rich import print as rprint

from .data.datamodule import DataModule
from .train.trainer import Trainer
from .infer.predictor import Predictor
from .explain.shap_explain import shap_explain_text

# Root app
app = typer.Typer(help="Toxic Content Detection Toolkit CLI")

# Data group for subcommands
data_app = typer.Typer(help="Dataset utilities: download, prepare, inspect")
app.add_typer(data_app, name="data")


# ============================================================
# DATA SUBCOMMANDS
# ============================================================

@data_app.command("prepare")
def data_prepare(
    dataset: str = typer.Option("jigsaw", help="Dataset name: 'jigsaw' or 'hatexplain'"),
    out: pathlib.Path = typer.Option(pathlib.Path("data") / "prepared", help="Output directory"),
):
    """
    Download and preprocess a dataset (HateXplain or Jigsaw Toxic Comments).
    """
    dm = DataModule.from_name(dataset, out_dir=out)
    dm.prepare()
    rprint(f"[green]✅ Prepared {dataset} dataset at {out}[/green]")


# ============================================================
# TRAIN COMMAND
# ============================================================

@app.command("train")
def train(
    dataset: str = "jigsaw",
    model: str = "bert-base-uncased",
    epochs: int = 1,
    output: pathlib.Path = pathlib.Path("runs") / "bert_model",
    batch_size: int = 16,
    lr: float = 2e-5,
    labels: Optional[str] = None,
):
    """
    Train a multi-label classifier on a supported dataset.
    """
    dm = DataModule.from_name(dataset)

    # Label defaults depending on dataset
    if labels:
        label_list = labels.split(",")
    elif dataset.lower() == "jigsaw":
        label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    else:
        label_list = ["hate", "harassment", "misinformation", "spam"]

    trainer = Trainer(model_name=model, labels=label_list, out_dir=output)
    trainer.fit(dm, epochs=epochs, batch_size=batch_size, lr=lr)
    rprint(f"[green]✅ Training complete. Model saved to {output}[/green]")


# ============================================================
# EVAL COMMAND
# ============================================================

@app.command("eval")
def evaluate(
    run: pathlib.Path = typer.Argument(..., help="Run/checkpoint directory"),
    split: str = "validation",
    threshold: float = 0.5,
):
    """
    Evaluate a trained checkpoint on a given split.
    """
    pred = Predictor.from_run(run)
    metrics = pred.evaluate(split=split, threshold=threshold)
    rprint(metrics)


# ============================================================
# INFER COMMAND
# ============================================================

@app.command("infer")
def infer(
    run: pathlib.Path = typer.Argument(..., help="Run/checkpoint directory"),
    threshold: float = 0.5,
):
    """
    Run inference from stdin (pipe or echo a text string).
    Example:
        echo "You are terrible!" | toxdet infer runs/bert_model
    """
    pred = Predictor.from_run(run)
    text = sys.stdin.read().strip()
    if not text:
        rprint("[red]No input text provided![/red]")
        raise typer.Exit(1)
    result = pred.predict([text], threshold=threshold)[0]
    rprint(json.dumps(result, ensure_ascii=False, indent=2))


# ============================================================
# EXPLAIN COMMAND
# ============================================================

@app.command("explain")
def explain(
    method: str = "shap",
    run: pathlib.Path = pathlib.Path("runs/bert_model"),
    text: str = "Example toxic text",
):
    """
    Explain model predictions for a given text using SHAP or LIME.
    """
    if method.lower() == "shap":
        shap_explain_text(run, text)
    else:
        rprint("[yellow]Only SHAP demo implemented in CLI for now.[/yellow]")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    app()
