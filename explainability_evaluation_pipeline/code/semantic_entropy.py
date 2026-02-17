import requests
import json
from config import ENTAILMENT_THRESHOLD, SampleSequencesFlags, CORRECTNESS_THRESHOLD
# from prolog_graphrag_pipeline import main_driver
import math
import csv
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"
TEST_DIR = ROOT / "test_cases"
LOGS_DIR = ROOT / "logs/prolog_graphrag_pipeline_logs"

import requests
import json
from config import ENTAILMENT_THRESHOLD, SampleSequencesFlags
# from prolog_graphrag_pipeline import main_driver
import math
import csv

API_URL = "https://api-inference.huggingface.co/models/microsoft/deberta-large-mnli"
headers = {"Authorization": "Bearer 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"}

def evaluate(prompt: str, answer: str) -> float:
    return 0.5

def sample_sequences(prompt: str, flag: str) -> dict:
    # return main_driver.run_pipeline(prompt=prompt, flag=flag, sample_mode=True)
    return None

def query_deberta(seq1: str, seq2: str):
    input = {
        "inputs": f"{seq1} [SEP] {seq2}",
    }
    judgment = requests.post(API_URL, headers=headers, json=input)
    return judgment.json()
    
def check_entailment(seq1: str, seq2: str) -> bool:
    entailment_1 = query_deberta(seq1, seq2)
    entailment_2 = query_deberta(seq2, seq1)
    return entailment_1 >= ENTAILMENT_THRESHOLD and entailment_2 >= ENTAILMENT_THRESHOLD

def cluster_sequences(sequences: list[dict]) -> list[list[dict]]:
    C = [[sequences[0]]]
    M = 5
    for m in range(1, M):
        assigned = False
        for c in C:
            first_seq = c[0]["text_answer"]
            entailment = check_entailment(first_seq, sequences[m]["text_answer"])
            if entailment:
                assigned = True
                c.append(sequences[m]["text_answer"])
                break
        if not assigned:
            C.append([sequences[m]["text_answer"]])
    return C

def compute_likelihood(cluster: list[dict]) -> float:
    likelihood = 0
    for seq in cluster:
        likelihood += compute_aggregated_logprobs(seq)
    return likelihood

def compute_aggregated_logprobs(response: dict) -> float:
    sequence_probability = 0
    for token_logprob in response["logprobs"]:
        sequence_probability += token_logprob.logprob
        
    sequence_probability = math.e ** sequence_probability
    return sequence_probability
    

def compute_semantic_entropy(test_case_id: int, prompt: str):
    semantic_entropies = {
        r"x": None,
        r"q": None,
        r"x\c": None
    }
    output_sequences = {
        "se_x": {
            "prompt": None,
            "sequences": [],
            "se": None
            },
        "se_q": {
            "prompt": None,
            "sequences": [],
            "se": None
            },
        "se_x_c": {
            "prompt": None,
            "sequences": [],
            "se": None
            }
    }
    for flag in list(SampleSequencesFlags):
        output = sample_sequences(prompt=prompt, flag=flag.value[1])
        reconstructed_prompt: str = output["prompt"]
        output_sequences["se_" + flag.value[1]]["prompt"] = reconstructed_prompt
        sequences = output["sequences"]
        for i, seq in enumerate(sequences):
            score = evaluate(prompt=reconstructed_prompt, answer=seq["text_answer"])
            d = {f"se_{flag.value[1]}_sequence_{i}": seq["text_answer"], f"se_{flag.value[1]}_sequence_{i}_label": score}
            output_sequences["se_" + flag.value[1]]["sequences"].append(d)
        clusters = cluster_sequences(sequences)
        semantic_entropy = 0
        sum_of_cluster_probabilities = sum(compute_likelihood(cluster) for cluster in clusters)
        for cluster in clusters:
            p_Ci_x = compute_likelihood(cluster) / sum_of_cluster_probabilities
            semantic_entropy += p_Ci_x * math.log(p_Ci_x)
        semantic_entropy *= -1
        semantic_entropies[flag.value[1]] = semantic_entropy
        output_sequences["se_" + flag.value[1]]["se"] = semantic_entropy
        
    for key in semantic_entropies.keys():
        if not semantic_entropies[key]:
            raise ValueError(f"Semantic entropy for {key} is missing.")
        
    log_result(test_case_id=test_case_id, semantic_entropies=semantic_entropies)
    
def run_semantic_entropy_computations():
    with open(TEST_DIR / "explainability_test_cases.jsonl", "r") as f:
        for line in f:
            test_case = json.loads(line)
            test_case_id = test_case["test_case_id"]
            prompt = test_case["question"]
        
            compute_semantic_entropy(test_case_id=test_case_id, prompt=prompt)
    
def log_result(test_case_id: int, semantic_entropies: dict) -> None:
    with open(IO_DIR / "semantic_entropy.csv", 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case_id", "se_q", "se_q_label", "se_x", "se_x_label", "se_x_c", "se_x_c_label"]
        )
        writer.writerow({
            "test_case_id": test_case_id,
            "se_q": semantic_entropies[r"q"],
            "se_q_label": int(semantic_entropies[r"q"] < CORRECTNESS_THRESHOLD),
            "se_x": semantic_entropies[r"x"],
            "se_x_label": int(semantic_entropies[r"x"] < CORRECTNESS_THRESHOLD),
            "se_x_c": semantic_entropies[r"x\c"],
            "se_x_c_label": int(semantic_entropies[r"x\c"] < CORRECTNESS_THRESHOLD)
        })
    


def log_sequences(test_case_id: int, output_sequences: dict) -> None:
    row = dict()
    
    row["test_case_id"] = test_case_id
    for flag in list(SampleSequencesFlags):
        row[f"se_{flag.value[1]}_prompt"] = output_sequences[f"se_{flag.value[1]}"]["prompt"]
        sequences = output_sequences[f"se_{flag.value[1]}"]["sequences"]
        for i, seq in enumerate(sequences):
            row[f"se_{flag.value[1]}_sequence_{i+1}"] = seq[f"se_{flag.value[1]}_sequence_{i+1}"]
            row[f"se_{flag.value[1]}_sequence_{i+1}_label"] = seq[f"se_{flag.value[1]}_sequence_{i+1}_label"]
        row[f"se_{flag.value[1]}"] = output_sequences[f"se_{flag.value[1]}"]["se"]
        
    
    with open(LOGS_DIR / "explainability.jsonl", 'a', newline='') as f:
        f.write(json.dumps(row) + "\n")
        