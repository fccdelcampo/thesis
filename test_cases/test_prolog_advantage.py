import os
import sys
import time
import json
import pandas as pd

# Add thesis root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag
try:
    from tuned_pipeline.main_driver import run_pipeline as run_tuned_pipeline
except ImportError:
    print("Warning: Could not import tuned_pipeline. Using naive ollama call.")
    import ollama
    def run_tuned_pipeline(q):
        try:
            res = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': q}])
            return {"answer": res['message']['content'], "contexts": []}
        except:
            return {"answer": "Error", "contexts": []}

from hallucination_evaluation_pipeline.code.mmlu import extract_answer_letter, score_answer

RESULTS_CSV = os.path.join(project_root, "hallucination_evaluation_results", "curated_prolog_advantage.csv")

# 5 Questions designed to break an 8B model's parametric memory but solvable via KBPedia semantic logic
# 1. Obscure Taxonomic Logic
# 2. Transitive Geographical Logic
# 3. Medical Fallacy Detection
# 4. Deep historical ontology
# 5. Complex biochemical relationship
TEST_CASES = [
    {
        "id": 1,
        "question": "Which of these organisms belongs to the order Strigiformes and is characterized by a facial disk? A. Buteo jamaicensis B. Tyto alba C. Falco peregrinus D. Corvus corax",
        "answer": "B" # Barn owl (Tyto alba) is Strigiformes. The others are Accipitriformes, Falconiformes, Passeriformes.
    },
    {
        "id": 2,
        "question": "If a disease is endemic to the geographic region bordered by the Andes to the west and the Atlantic Ocean to the east, which of these diseases is most likely to be naturally found there? A. Sleeping Sickness (Trypanosomiasis) B. Lyme Disease C. Chagas Disease D. Malaria",
        "answer": "C" # South America -> Chagas Disease
    },
    {
        "id": 3,
        "question": "An argument claims that a specific diet must be healthy because it has been used by a nomadic tribe for 2,000 years. Which logical fallacy is this an example of? A. Post hoc ergo propter hoc B. Appeal to tradition C. Straw man D. Appeal to common practice",
        "answer": "B"
    },
    {
        "id": 4,
        "question": "The Treaty of Tordesillas divided newly discovered lands outside Europe between which two empires? A. Spanish Empire and Portuguese Empire B. British Empire and French Empire C. Ottoman Empire and Russian Empire D. Dutch Empire and Spanish Empire",
        "answer": "A"
    },
    {
        "id": 5,
        "question": "Which enzyme acts to break down hydrogen peroxide into water and oxygen in eukaryotic cells? A. Amylase B. Catalase C. Helicase D. Polymerase",
        "answer": "B"
    }
]

def format_question(tc: dict) -> str:
    # Ensure options are formatted correctly as standard LLM evaluations usually do
    return f"{tc['question']}\nAnswer with the letter only (A, B, C, or D)."

def run_test():
    print("================================================================================")
    print("STARTING CURATED PROLOG GRAPHRAG ADVANTAGE BENCHMARK")
    print("================================================================================\n")
    
    results = []
    
    for tc in TEST_CASES:
        print(f"--- Question {tc['id']} ---")
        prompt = format_question(tc)
        print(f"Q: {prompt.split(chr(10))[0]}")
        print(f"Expected: {tc['answer']}")
        
        # 1. Tuned Pipeline (Parametric Memory Only)
        print("  Evaluating Tuned LLM...")
        t0 = time.time()
        res_tuned = run_tuned_pipeline(prompt)
        time_tuned = time.time() - t0
        let_tuned = extract_answer_letter(res_tuned.get("answer", ""))
        score_tuned = score_answer(let_tuned, tc['answer'])
        
        # 2. Prolog GraphRAG (KBPedia Symbolic Logic)
        print("  Evaluating Prolog GraphRAG...")
        t0 = time.time()
        res_prolog = run_prolog_graphrag(prompt, flag="prolog")
        time_prolog = time.time() - t0
        let_prolog = extract_answer_letter(res_prolog.get("answer", ""))
        score_prolog = score_answer(let_prolog, tc['answer'])
        
        print("\n  RESULTS:")
        print(f"  Tuned LLM : Predicted [{let_tuned}] ({score_tuned}) | Time: {time_tuned:.2f}s")
        print(f"  Prolog    : Predicted [{let_prolog}] ({score_prolog}) | Time: {time_prolog:.2f}s")
        print("--------------------------------------------------------------------------------\n")
        
        results.append({
            "question_id": tc["id"],
            "question": tc["question"],
            "correct_answer": tc["answer"],
            "tuned_predicted": let_tuned,
            "tuned_score": score_tuned,
            "tuned_time_seconds": round(time_tuned, 2),
            "prolog_predicted": let_prolog,
            "prolog_score": score_prolog,
            "prolog_time_seconds": round(time_prolog, 2)
        })
        
    df = pd.DataFrame(results)
    
    # Calculate Summaries
    tuned_accuracy = (df["tuned_score"].sum() / len(df)) * 100
    prolog_accuracy = (df["prolog_score"].sum() / len(df)) * 100
    tuned_avg_time = df["tuned_time_seconds"].mean()
    prolog_avg_time = df["prolog_time_seconds"].mean()
    
    print("================================================================================")
    print("FINAL SUMMARY")
    print("================================================================================")
    print(f"Accuracy - Tuned LLM       : {tuned_accuracy}%")
    print(f"Accuracy - Prolog GraphRAG : {prolog_accuracy}%")
    print(f"Latency  - Tuned LLM       : {tuned_avg_time:.2f}s average")
    print(f"Latency  - Prolog GraphRAG : {prolog_avg_time:.2f}s average")
    
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nDetailed results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    run_test()
