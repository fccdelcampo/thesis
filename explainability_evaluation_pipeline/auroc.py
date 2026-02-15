def create_judgment_list(semantic_entropies: list[float], threshold: float):
    return [int(entropy >= threshold) for entropy in semantic_entropies]
