import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mech_interp_utils import MODEL_NAME, answer_token_id, model_device_inputs, model_load_kwargs, role_margin


def run_causal_tracing(max_pairs=None):
    """Patch only the final prediction-site residual, with a role reversal control."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_load_kwargs())
    model.eval()
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        dataset = json.load(f)
    pairs = defaultdict(dict)
    for item in dataset:
        pairs[item["pair_id"]][item["condition"]] = item
    pair_ids = sorted(pairs)
    if max_pairs is not None:
        pair_ids = pair_ids[::max(1, len(pair_ids) // max_pairs)][:max_pairs]

    effects = []
    for layer in range(model.config.num_hidden_layers):
        layer_effects = []
        for pair_id in pair_ids:
            # Evaluate both syntactic forms; forward/reversed retain identical
            # token lengths and lexical content while reversing the answer.
            for source_key, target_key in (("active_forward", "active_reversed"), ("passive_forward", "passive_reversed")):
                source, target = pairs[pair_id][source_key], pairs[pair_id][target_key]
                source_inputs = model_device_inputs(tokenizer, source["prompt"])
                target_inputs = model_device_inputs(tokenizer, target["prompt"])
                if source_inputs.input_ids.shape != target_inputs.input_ids.shape:
                    raise ValueError("Patch pairs must have matched prompt lengths")
                correct = answer_token_id(tokenizer, target["prompt"], target["agent"])
                incorrect = answer_token_id(tokenizer, target["prompt"], target["patient"])
                with torch.no_grad():
                    source_hidden = model(**source_inputs, output_hidden_states=True).hidden_states[layer + 1][0, -1].clone()
                    base_logits = model(**target_inputs).logits[0, -1]
                base_margin = role_margin(base_logits, correct, incorrect)

                def patch_final(module, inputs, output):
                    hidden = output[0].clone()
                    hidden[0, -1] = source_hidden
                    return (hidden, *output[1:])

                handle = model.model.layers[layer].register_forward_hook(patch_final)
                try:
                    with torch.no_grad():
                        patched_logits = model(**target_inputs).logits[0, -1]
                finally:
                    handle.remove()
                layer_effects.append((role_margin(patched_logits, correct, incorrect) - base_margin).item())
        effects.append(sum(layer_effects) / len(layer_effects))
        print(f"Layer {layer:2}: mean target-margin change {effects[-1]:+.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(effects, marker="o")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Layer"); plt.ylabel("Change in target answer logit margin")
    plt.title("Final-site Residual Patching (active and passive controls)")
    (ROOT / "results").mkdir(exist_ok=True)
    plt.savefig(ROOT / "results" / "activation_patching_effect.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    run_causal_tracing()
