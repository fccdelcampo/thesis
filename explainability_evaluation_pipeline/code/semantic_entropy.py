import requests
import json
from config import ENTAILMENT_THRESHOLD, SampleSequencesFlags, CORRECTNESS_THRESHOLD
# from prolog_graphrag_pipeline import main_driver
import math
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"
TEST_DIR = ROOT / "test_cases"

import requests
import json
from config import ENTAILMENT_THRESHOLD, SampleSequencesFlags
# from prolog_graphrag_pipeline import main_driver
import math
import csv

API_URL = "https://api-inference.huggingface.co/models/microsoft/deberta-large-mnli"
headers = {"Authorization": "Bearer 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"}

def sample_sequences(prompt: str, flag: str) -> list[dict]:
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
    output_sequences = list()
    for flag in list(SampleSequencesFlags):
        sequences = sample_sequences(prompt=prompt, flag=flag.value)
        output_sequences += [a["text_answer"] for a in sequences]
        clusters = cluster_sequences(sequences)
        semantic_entropy = 0
        sum_of_cluster_probabilities = sum(compute_likelihood(cluster) for cluster in clusters)
        for cluster in clusters:
            p_Ci_x = compute_likelihood(cluster) / sum_of_cluster_probabilities
            semantic_entropy += p_Ci_x * math.log(p_Ci_x)
        semantic_entropy *= -1
        semantic_entropies[flag.value] = semantic_entropy
        
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
