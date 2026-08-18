import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def extract_activations(sample_size=200):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    
    # We load in float16 to save RAM on the Jetson
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    
    with open(ROOT / "data" / "aligned_dataset.json", "r") as f:
        dataset = json.load(f)
        
    if sample_size:
        dataset = dataset[:sample_size]
        
    # Dictionary to store the output of our hooks
    activations_buffer = {}
    
    # The hook function that PyTorch will call after every layer
    def get_activation(layer_name):
        def hook(module, input, output):
            # output[0] contains the hidden states: shape (batch, seq_len, hidden_dim)
            # We detach and move to CPU immediately to prevent VRAM overflow
            activations_buffer[layer_name] = output[0].detach().cpu()
        return hook
        
    handles = []
    num_layers = model.config.num_hidden_layers
    
    # Register a forward hook on the residual stream of every layer
    for i in range(num_layers):
        # In the Qwen architecture, the transformer layers are in `model.model.layers`
        layer = model.model.layers[i]
        handle = layer.register_forward_hook(get_activation(f"layer_{i}"))
        handles.append(handle)
        
    print(f"\nExtracting activations for {len(dataset)} sentences across {num_layers} layers...")
    os.makedirs(ROOT / "results", exist_ok=True)
    
    extracted_data = []
    
    for idx, item in enumerate(dataset):
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        
        # Run the forward pass (hooks will automatically populate activations_buffer)
        with torch.no_grad():
            _ = model(input_ids)
            
        # Extract only the specific tokens we care about
        item_tensors = {
            "agent": {},
            "patient": {},
            "verb": {},
            "final": {}
        }
        
        for i in range(num_layers):
            # Remove batch dimension
            layer_acts = activations_buffer[f"layer_{i}"][0] 
            
            # Clone to ensure we don't hold references to the entire computation graph
            item_tensors["agent"][i] = layer_acts[item["extract_agent"]].clone()
            item_tensors["patient"][i] = layer_acts[item["extract_patient"]].clone()
            item_tensors["verb"][i] = layer_acts[item["extract_verb"]].clone()
            item_tensors["final"][i] = layer_acts[item["extract_final"]].clone()
            
        extracted_data.append({
            "metadata": item,
            "activations": item_tensors
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(dataset)}")
            
    output_path = ROOT / "results" / "extracted_activations.pt"
    # Save using PyTorch's native format, which is heavily optimized for tensors
    torch.save(extracted_data, output_path)
    print(f"\nSuccessfully saved activations to {output_path}")
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    extract_activations(sample_size=None)
