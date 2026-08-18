import json
import itertools

def generate_balanced_minimal_pairs():
    # 1. Explicitly partition entities
    # Keep lexical generalisation separate from syntactic generalisation.
    # Every unordered pair is used, so each noun has exactly the same number
    # of agent and patient examples within its split.
    train_entities = ["dog", "cat", "boy", "girl", "lion", "mouse", "doctor", "artist"]
    test_entities = ["teacher", "student", "chef", "pilot"]

    verbs = ["chased", "helped", "followed", "watched", "avoided", "pushed"]

    # Use unordered pairs: add_quad supplies both role directions.  This gives
    # 672 train and 144 held-out examples (816 total) with six verbs.
    train_pairs = list(itertools.combinations(train_entities, 2))
    test_pairs = list(itertools.combinations(test_entities, 2))

    dataset = []
    pair_id = 0

    def add_quad(first_entity, second_entity, is_test_pair):
        nonlocal pair_id
        for verb_index, verb in enumerate(verbs):
            # Counterbalance which member of an unordered pair is the
            # ``forward`` agent. Without this, list order leaks into the
            # condition label (e.g. dog is always forward against cat).
            agent, patient = (first_entity, second_entity) if verb_index % 2 == 0 else (second_entity, first_entity)
            pair_id += 1
            
            # Active Forward
            dataset.append({
                "pair_id": pair_id, "condition": "active_forward",
                "sentence": f"The {agent} {verb} the {patient}.",
                "agent": agent, "patient": patient, "verb": verb,
                "subject": agent, "object": patient,
                "is_test_set": is_test_pair
            })
            # Active Reversed
            dataset.append({
                "pair_id": pair_id, "condition": "active_reversed",
                "sentence": f"The {patient} {verb} the {agent}.",
                "agent": patient, "patient": agent, "verb": verb,
                "subject": patient, "object": agent,
                "is_test_set": is_test_pair
            })
            # Passive Forward
            dataset.append({
                "pair_id": pair_id, "condition": "passive_forward",
                "sentence": f"The {patient} was {verb} by the {agent}.",
                "agent": agent, "patient": patient, "verb": verb,
                "subject": patient, "object": agent,
                "is_test_set": is_test_pair
            })
            # Passive Reversed
            dataset.append({
                "pair_id": pair_id, "condition": "passive_reversed",
                "sentence": f"The {agent} was {verb} by the {patient}.",
                "agent": patient, "patient": agent, "verb": verb,
                "subject": agent, "object": patient,
                "is_test_set": is_test_pair
            })

    # Build Train Sentences (672)
    for agent, patient in train_pairs:
        add_quad(agent, patient, is_test_pair=False)

    # Build Test Sentences (144)
    for agent, patient in test_pairs:
        add_quad(agent, patient, is_test_pair=True)

    return dataset

if __name__ == "__main__":
    data = generate_balanced_minimal_pairs()
    output_path = "data/minimal_pairs.json"
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    train_count = sum(1 for d in data if not d["is_test_set"])
    test_count = sum(1 for d in data if d["is_test_set"])
    
    print(f"Generated {len(data)} total balanced sentences.")
    print(f"  -> Train Set Sentences: {train_count}")
    print(f"  -> Test Set Sentences : {test_count}")
