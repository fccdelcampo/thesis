import comprehensiveness, auroc, sufficiency, semantic_entropy

def load_test_cases() -> list[int]:
    pass

def run_pipeline():
    test_case_ids = load_test_cases()
    semantic_entropy.run_semantic_entropy_computations(test_case_ids)
    auroc.run_auroc_computations()
    comprehensiveness.run_comprehensiveness_computations()
    sufficiency.run_sufficiency_computations()
