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

def run_cross_syntactic_probe():
    print("Loading extracted activations...")
    data = torch.load(ROOT / "results" / "extracted_activations.pt", weights_only=False) 
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        expected_examples = len(json.load(f))
    if len(data) != expected_examples:
        raise ValueError("Activation cache is stale; rerun extraction/activation_extraction.py after aligning data.")
    
    num_layers = len(data[0]["activations"]["agent"])
    print(f"Loaded {len(data)} sentences with {num_layers} layers of activations.")

    # Primary test isolates syntax: same vocabulary, active -> passive.
    # A separate held-out-vocabulary evaluation would otherwise confound the two.
    train_indices = [i for i, d in enumerate(data) if not d["metadata"]["is_test_set"] and "active" in d["metadata"]["condition"]]
    test_indices = [i for i, d in enumerate(data) if not d["metadata"]["is_test_set"] and "passive" in d["metadata"]["condition"]]

    print(f"\nCross-Syntactic Split:")
    print(f"Training on Active Sentences : {len(train_indices) * 2} vectors")
    print(f"Testing on Passive Sentences : {len(test_indices) * 2} vectors")

    layer_accuracies = []
    inverse_accuracies = []

    print("\nTraining layer-wise cross-syntactic probes...")
    print(f"{'Layer':<8} | {'Role Acc':<10} | {'Inverted Acc':<13}")
    print("-" * 43)

    for layer in range(num_layers):
        X_train, y_train = [], []
        X_test, y_test = [], []

        for idx in train_indices:
            acts = data[idx]["activations"]
            X_train.extend([acts["agent"][layer].numpy(), acts["patient"][layer].numpy()])
            y_train.extend([1, 0])

        for idx in test_indices:
            acts = data[idx]["activations"]
            X_test.extend([acts["agent"][layer].numpy(), acts["patient"][layer].numpy()])
            y_test.extend([1, 0])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # We increase regularization (C=0.1) to force the probe to find robust features
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
        clf.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        layer_accuracies.append(test_acc)
        inverse_acc = 1 - test_acc
        inverse_accuracies.append(inverse_acc)
        
        print(f"Layer {layer:<2} | {test_acc*100:>7.1f}% | {inverse_acc*100:>10.1f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_layers), layer_accuracies, marker='o', linewidth=2, color='#9467bd', label="Role-label accuracy")
    plt.plot(range(num_layers), inverse_accuracies, marker='x', linewidth=1.5, color='#d62728', label="Label-inverted accuracy")
    plt.axhline(0.5, color='grey', linestyle='--', alpha=0.7, label="Random Chance")
    
    plt.title("Cross-Syntax Role Probe Diagnostic (Train: Active, Test: Passive)", fontsize=14)
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel("Probe Test Accuracy on Passive Sentences", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.xticks(range(0, num_layers, 2))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/role_emergence_cross_syntactic.png", dpi=300, bbox_inches='tight')
    print("\nSaved emergence curve plot to: results/role_emergence_cross_syntactic.png")
    if max(inverse_accuracies) > 0.9 and max(layer_accuracies) < 0.1:
        print("WARNING: labels invert perfectly across voice. This probe is tracking a voice/position direction, not a voice-invariant semantic-role code.")

if __name__ == "__main__":
    run_cross_syntactic_probe()
