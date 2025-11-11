import shap
import numpy as np
import torch
from ..infer.predictor import Predictor

def shap_explain_text(run_dir, text: str):
    """
    Generate SHAP explanations for a single text input using a trained model.
    Requires that the model directory (run_dir) contains the model + tokenizer.
    """
    print(f"🔍 Explaining text with SHAP: {text[:80]}...")
    pred = Predictor.from_run(run_dir)
    tokenizer = pred.tokenizer

    # Define wrapper function that returns model probabilities
    def f(texts):
        outs = pred.predict(texts)
        return np.array([[o["scores"][lab] for lab in pred.labels] for o in outs])

    # SHAP text masker and explainer
    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer.sep_token or " "))
    shap_values = explainer([text])

    # Display in notebook or supported environments
    shap.plots.text(shap_values[0])
    print("✅ SHAP explanation generated.")
