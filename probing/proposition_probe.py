import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_proposition_probe():
    print("Loading extracted activations...")
    data = torch.load(ROOT / "results" / "extracted_activations.pt", weights_only=False) 
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        expected_examples = len(json.load(f))
    if len(data) != expected_examples:
        raise ValueError("Activation cache is stale; rerun extraction/activation_extraction.py after aligning data.")
    num_layers = len(data[0]["activations"]["final"])
    print(f"Loaded {len(data)} sentences. Probing the FINAL TOKEN across {num_layers} layers.")

    # Held-out nouns test lexical generalisation. This is a decodability
    # measurement, not by itself evidence that the representation is causal.
    train_indices = [i for i, d in enumerate(data) if not d["metadata"]["is_test_set"]]
    test_indices = [i for i, d in enumerate(data) if d["metadata"]["is_test_set"]]

    print(f"Training vectors (Train Nouns) : {len(train_indices)}")
    print(f"Testing vectors (Test Nouns)   : {len(test_indices)}")

    layer_accuracies = []

    for layer in range(num_layers):
        X_train, y_train = [], []
        X_test, y_test = [], []

        # Populate Train
        for idx in train_indices:
            acts = data[idx]["activations"]
            condition = data[idx]["metadata"]["condition"]
            
            X_train.append(acts["final"][layer].numpy())
            # Forward/reversed is balanced within active and passive syntax.
            y_train.append(1 if "forward" in condition else 0)

        # Populate Test
        for idx in test_indices:
            acts = data[idx]["activations"]
            condition = data[idx]["metadata"]["condition"]
            
            X_test.append(acts["final"][layer].numpy())
            y_test.append(1 if "forward" in condition else 0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train Probe on the Final Token
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
        clf.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        layer_accuracies.append(test_acc)
        
        if layer % 2 == 0:
            print(f"Layer {layer:<2} | Proposition Accuracy: {test_acc*100:>5.1f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_layers), layer_accuracies, marker='D', linewidth=2.5, color='#d62728', label="Proposition Probe (Final Token)")
    plt.axhline(0.5, color='grey', linestyle='--', alpha=0.7, label="Random Chance")
    
    plt.title("Decodability of Role Binding at the Final Prompt Token", fontsize=14)
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel("Test Accuracy (Unseen Nouns, Mixed Syntax)", fontsize=12)
    plt.ylim(0.4, 1.05)
    plt.xticks(range(0, num_layers, 2))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/proposition_emergence.png", dpi=300, bbox_inches='tight')
    print("\nSaved proposition emergence plot to: results/proposition_emergence.png")

if __name__ == "__main__":
    run_proposition_probe()
