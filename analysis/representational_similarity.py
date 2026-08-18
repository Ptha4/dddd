import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

def run_rsa():
    print("Loading extracted activations...")
    data = torch.load(ROOT / "results" / "extracted_activations.pt", weights_only=False)
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        expected_examples = len(json.load(f))
    if len(data) != expected_examples:
        raise ValueError("Activation cache is stale; rerun extraction/activation_extraction.py after aligning data.")
    
    num_layers = len(data[0]["activations"]["agent"])
    print(f"Loaded activations for {num_layers} layers.")

    # 1. Group activations by Word and Role
    # Dictionary structure: groupings[layer][word][role] = list of tensors
    groupings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # To keep the analysis crisp, we will average across two highly frequent entities.
    # You can change these to any two entities present in your dataset.
    target_words = ["dog", "cat"]
    
    for item in data:
        meta = item["metadata"]
        acts = item["activations"]
        
        agent_word = meta["agent"]
        patient_word = meta["patient"]
        
        for layer in range(num_layers):
            if agent_word in target_words:
                groupings[layer][agent_word]["agent"].append(acts["agent"][layer])
            if patient_word in target_words:
                groupings[layer][patient_word]["patient"].append(acts["patient"][layer])

    # 2. Compute Mean Representations and Track Similarities
    lexical_similarities = []
    role_similarities = []
    
    print("\nComputing Cosine Similarities...")
    print(f"{'Layer':<8} | {'Lexical Sim':<15} | {'Role Sim':<15}")
    print("-" * 44)

    for layer in range(num_layers):
        layer_data = groupings[layer]
        
        # Calculate the mean vector (centroid) for each state
        # e.g., the "average dog acting as an agent"
        mean_reps = {}
        for word in target_words:
            for role in ["agent", "patient"]:
                tensors = layer_data[word].get(role, [])
                if tensors:
                    mean_reps[f"{word}_{role}"] = torch.stack(tensors).mean(dim=0)
                else:
                    print(f"Warning: Missing data for {word} as {role}")

        # Compute Lexical Similarity: Same concept, different role
        # (DOG-Agent vs DOG-Patient) & (CAT-Agent vs CAT-Patient)
        lex_sim_dog = F.cosine_similarity(mean_reps["dog_agent"].unsqueeze(0), 
                                          mean_reps["dog_patient"].unsqueeze(0)).item()
        lex_sim_cat = F.cosine_similarity(mean_reps["cat_agent"].unsqueeze(0), 
                                          mean_reps["cat_patient"].unsqueeze(0)).item()
        avg_lexical_sim = (lex_sim_dog + lex_sim_cat) / 2
        
        # Compute Role Similarity: Different concept, same role
        # (DOG-Agent vs CAT-Agent) & (DOG-Patient vs CAT-Patient)
        role_sim_agent = F.cosine_similarity(mean_reps["dog_agent"].unsqueeze(0), 
                                             mean_reps["cat_agent"].unsqueeze(0)).item()
        role_sim_patient = F.cosine_similarity(mean_reps["dog_patient"].unsqueeze(0), 
                                               mean_reps["cat_patient"].unsqueeze(0)).item()
        avg_role_sim = (role_sim_agent + role_sim_patient) / 2
        
        lexical_similarities.append(avg_lexical_sim)
        role_similarities.append(avg_role_sim)
        
        print(f"Layer {layer:<2} | {avg_lexical_sim:>13.3f}   | {avg_role_sim:>13.3f}")

    # 3. Plot the RSA Trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_layers), lexical_similarities, marker='o', linewidth=2.5, 
             color='#1f77b4', label='Lexical Similarity (Same Noun, Different Role)')
    plt.plot(range(num_layers), role_similarities, marker='s', linewidth=2.5, 
             color='#ff7f0e', label='Role Similarity (Different Noun, Same Role)')
    
    plt.title("Representational Geometry: Lexical vs. Role Similarity", fontsize=14)
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xticks(range(0, num_layers, 2))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc="best")
    
    os.makedirs(ROOT / "results", exist_ok=True)
    plot_path = ROOT / "results" / "rsa_geometry_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved RSA trajectory plot to: {plot_path}")

if __name__ == "__main__":
    run_rsa()
