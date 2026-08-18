import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mech_interp_utils import MODEL_NAME, build_prompt


def align_tokens():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    with open(ROOT / "data" / "minimal_pairs.json") as f:
        dataset = json.load(f)

    aligned_dataset = []
    for item in dataset:
        sentence = item["sentence"]
        prompt = build_prompt(tokenizer, sentence)
        sentence_start = prompt.index(sentence)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)

        def token_indices(word):
            match = re.search(r"\b" + re.escape(word) + r"\b", sentence)
            if not match:
                raise ValueError(f"{word!r} not found in {sentence!r}")
            start, end = match.span()
            indices = {encoded.char_to_token(sentence_start + i) for i in range(start, end)}
            indices.discard(None)
            if not indices:
                raise ValueError(f"Could not align {word!r} in prompt")
            return sorted(indices)

        agent_span, patient_span, verb_span = map(token_indices, (item["agent"], item["patient"], item["verb"]))
        aligned_dataset.append(item | {
            "prompt": prompt,
            "input_ids": encoded["input_ids"],
            "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            "agent_span": agent_span,
            "patient_span": patient_span,
            "verb_span": verb_span,
            "extract_agent": agent_span[-1],
            "extract_patient": patient_span[-1],
            "extract_verb": verb_span[-1],
            # This is the next-token prediction site, not merely a period.
            "extract_final": len(encoded["input_ids"]) - 1,
        })

    with open(ROOT / "data" / "aligned_dataset.json", "w") as f:
        json.dump(aligned_dataset, f, indent=2)
    print(f"Aligned {len(aligned_dataset)} prompt-formatted examples.")

if __name__ == "__main__":
    align_tokens()
