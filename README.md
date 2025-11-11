# Toxic Content Detection Toolkit

Open-source, modular pipeline for detecting and classifying toxic text across multiple domains.

## Key Features
- Multi-label classification: hate, harassment, misinformation, spam (extensible)
- Dataset adapters: HateXplain, Civil Comments (Jigsaw replacement), Reddit — pluggable interface
- Pretrained checkpoints: BERT/DistilBERT fine-tuned (community-hostable)
- Visualization dashboard: toxicity heatmaps by topic/community
- Explainability: SHAP or LIME for model interpretability
- Batteries included: CLI, config, evaluation, unit tests, CI, model cards

## Quickstart

```bash
# 1. Install
pip3 install -e .

# 2. Explore the CLI
toxdet --help

# 3. Download and preprocess a dataset
# You can choose one of the supported datasets:
# - jigsaw → uses the Civil Comments dataset (recommended)
# - hatexplain → optional, requires manual download
toxdet data prepare --dataset jigsaw --out data/jigsaw

# 4. Train a multi-label model
toxdet train --dataset jigsaw --model bert-base-uncased --epochs 1 --output runs/bert_cc

# 5. Evaluate
toxdet eval runs/bert_cc --split validation

# 6. Inference (JSONL in, JSON out)
echo '{"text": "You are a terrible person"}' | toxdet infer runs/bert_cc --threshold 0.5

# 7. Explain a prediction with SHAP
toxdet explain shap --run runs/bert_cc --text "Example text to explain"
