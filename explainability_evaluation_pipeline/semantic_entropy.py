import requests

API_URL = "https://api-inference.huggingface.co/models/microsoft/deberta-large-mnli"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def sample_sequences(prompt: str) -> list[str]:
    pass

def query_deberta(seq1: str, seq2: str):
    input = {
        "inputs": f"{seq1} [SEP] {seq2}",
    }
    judgment = requests.post(API_URL, headers=headers, json=input)
    return judgment.json()
    
def check_entailment(seq1: str, seq2: str) -> bool:
    entailment_1 = query_deberta(seq1, seq2)
    entailment_2 = query_deberta(seq2, seq1)
    return entailment_1 > ENTAILMENT_THRESHOLD and entailment_2 > ENTAILMENT_THRESHOLD

def cluster_sequences(sequences: list[str]) -> list[list[str]]:
    C = [[sequences[0]]]
    M = 5
    for m in range(2, M):
        for c in C:
            first_seq = c[0]
            entailment = check_entailment(first_seq, sequences[m])
            if entailment:
                c.append(sequences[m])
        C.append([sequences[m]])
    return C

def compute_likelihood(cluster: list[str]) -> float:
    likelihood = 0
    for seq in cluster:
        likelihood += compute_aggregated_logprobs(seq)
    return likelihood

def compute_aggregated_logprobs(dict) -> float:
    pass

def compute_semantic_entropy(test_case_id: int, prompt: str) -> float:
    sequences = sample_sequences(prompt)
    clusters = cluster_sequences(sequences)
    semantic_entropy = 0
    for cluster in clusters:
        semantic_entropy += compute_likelihood(cluster)
    semantic_entropy *= -1
    log_result(test_case_id, semantic_entropy)
    
def log_result(test_case_id: int, semantic_entropy: float) -> None:
    pass
    