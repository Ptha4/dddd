import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mech_interp_utils import DEVICE, MODEL_NAME, balanced_subset, model_load_kwargs


def analyze_attention_heads(sample_size=None, top_k=10):
    """Rank heads by final-prediction-site preference for agent over patient.

    The query is the final prompt token, which can causally attend to both
    entities. This is descriptive head selection, not evidence of causality.
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_load_kwargs())
    model.eval()
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        dataset = json.load(f)
    dataset = balanced_subset(dataset, sample_size)

    scores = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads))
    for item in dataset:
        input_ids = torch.tensor([item["input_ids"]], device=DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        query = item["extract_final"]
        for layer, layer_attention in enumerate(outputs.attentions):
            heads = layer_attention[0]
            agent = heads[:, query, item["agent_span"]].sum(dim=-1)
            patient = heads[:, query, item["patient_span"]].sum(dim=-1)
            scores[layer] += (agent - patient).float().cpu().numpy()
    scores /= len(dataset)

    plt.figure(figsize=(14, 8))
    vmax = max(np.abs(scores).max(), 1e-8)
    plt.imshow(scores, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(label="Final-token attention: agent minus patient")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title("Descriptive Agent/Patient Attention Preference")
    (ROOT / "results").mkdir(exist_ok=True)
    plt.savefig(ROOT / "results" / "attention_head_bias.png", dpi=300, bbox_inches="tight")

    indices = np.argsort(np.abs(scores).ravel())[-top_k:][::-1]
    candidates = [{"layer": int(i // scores.shape[1]), "head": int(i % scores.shape[1]), "score": float(scores.ravel()[i])} for i in indices]
    with open(ROOT / "results" / "attention_head_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    print("Saved descriptive candidates; validate them with ablation:")
    for candidate in candidates:
        print(candidate)


if __name__ == "__main__":
    analyze_attention_heads()
