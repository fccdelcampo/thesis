import csv
import os
import sys
import time
import math
import requests
import traceback

# Ensure project root is on the path so llm_config can be imported
_project_root = os.path.abspath(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME

# Metrics
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
    from ragas import evaluate
    from datasets import Dataset
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
        from ragas import evaluate
        from datasets import Dataset
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError as e:
        print(f"Warning: ragas not installed or import failed: {e}. Evaluation logic will simulate calls or fail.")

def unload_ollama_model():
    """Force-unload the model from VRAM so RAGAS can use it without OOM."""
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": MODEL_NAME,
            "prompt": "",
            "keep_alive": 0
        }, timeout=10)
        print("  [VRAM] Model unloaded successfully.")
    except Exception as e:
        print(f"  [VRAM] Warning: Could not unload model: {e}")

class CleanOutputChatOllama(ChatOllama):
    """Wrapper that strips non-JSON characters from LLM responses so RAGAS can parse them.
    
    Overrides _generate (which invoke/generate call internally) to ensure
    all code paths produce clean JSON output.
    """
    
    @staticmethod
    def _clean_json(text: str) -> str:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1]
        # Try array format too (some RAGAS prompts expect arrays)
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1]
        return text

    def _clean_result(self, result):
        """Clean all message contents in a ChatResult."""
        if hasattr(result, 'generations'):
            for gen_list in result.generations:
                for gen in gen_list if isinstance(gen_list, list) else [gen_list]:
                    if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                        gen.message.content = self._clean_json(gen.message.content)
                    if hasattr(gen, 'text'):
                        gen.text = self._clean_json(gen.text)
        return result

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        # ChatResult has .generations list
        if hasattr(result, 'generations'):
            for gen in result.generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_json(gen.message.content)
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        return result

    async def _agenerate(self, *args, **kwargs):
        result = await super()._agenerate(*args, **kwargs)
        if hasattr(result, 'generations'):
            for gen in result.generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_json(gen.message.content)
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        return result

    def invoke(self, *args, **kwargs):
        result = super().invoke(*args, **kwargs)
        if hasattr(result, 'content'):
            result.content = self._clean_json(result.content)
        return result

    async def ainvoke(self, *args, **kwargs):
        result = await super().ainvoke(*args, **kwargs)
        if hasattr(result, 'content'):
            result.content = self._clean_json(result.content)
        return result

# Loaders
from test_cases.loaders.mmlu_loader import MMLULoader
from test_cases.loaders.ai2_arc_loader import AI2_ARC_Loader

import sys

# Ensure the pipeline modules can be imported
pipeline_dir = os.path.join(os.path.dirname(__file__), 'prolog_graphrag_pipeline')
sys.path.append(pipeline_dir)

# Import the REAL pipeline driver (Full Prolog GraphRAG)
try:
    from prolog_graphrag_pipeline import main_driver
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import main_driver: {e}")
    sys.exit(1)

class GraphRAGSystem:
    def query(self, input_text):
        try:
            # Run the actual pipeline (Prolog GraphRAG)
            # flag="" means run everything (GraphRAG -> Prolog -> LLM)
            return main_driver.run_pipeline(input_text, flag="", include_conclusion=True)
        except Exception as e:
            error_msg = f"Error querying pipeline: {e}\n{traceback.format_exc()}"
            print(error_msg)
            # Write to error log
            with open("eval_error.log", "a") as f: # Append mode, not write
                f.write(error_msg + "\n" + "="*50 + "\n")
            return {
                "answer": "Error",
                "contexts": [],
                "error": str(e)
            }

def sanitize(text):
    if isinstance(text, str):
        return text.replace('\n', ' ').strip()
    return text

def run_evaluation():
    # 1. Setup
    graph_rag = GraphRAGSystem()
    loaders = [
        MMLULoader(),
        AI2_ARC_Loader(),
    ]
    
    results_dir = "test_cases/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "evaluation_log.csv")
    
    # CSV Header
    fieldnames = [
        "dataset", "question", "ground_truth", "model_answer", 
        "faithfulness_score", "answer_relevancy_score", 
        "context_recall_score", "mmlu_accuracy", "execution_time"
    ]

    print("Starting Evaluation Pipeline...")
    
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for loader in loaders:
            dataset_name = loader.__class__.__name__.replace("Loader", "")
            print(f"\nLoading {dataset_name}...")
            test_cases = loader.load_data()
            
            for i, case in enumerate(test_cases[:1]):
                print(f"  Processing {i+1}/{len(test_cases)}: {case['question']}...")
                
                # 2. Query System
                start_time = time.time()
                response = graph_rag.query(case['formatted_input'])
                duration = time.time() - start_time
                
                answer = response.get('answer', '')
                contexts = response.get('contexts', [])
                
                # If the pipeline failed, skip Ragas to avoid confusing metrics (like 0.09 context recall on empty context)
                if answer == "Error":
                    print(f"  Skipping Ragas for {case['question']} due to pipeline error.")
                    row = {
                        "dataset": case['dataset'],
                        "question": sanitize(case['question']),
                        "ground_truth": sanitize(case['ground_truth']),
                        "model_answer": "Error",
                        "faithfulness_score": "NaN",
                        "answer_relevancy_score": "NaN",
                        "context_recall_score": "NaN",
                        "mmlu_accuracy": 0.0,
                        "execution_time": duration
                    }
                    writer.writerow(row)
                    continue

                # Unload GraphRAG model from VRAM before RAGAS evaluation
                unload_ollama_model()
                
                # Sanitize fields by removing newlines

                # 4. Batched Evaluation Setup
                # Ragas works best when evaluating a dataset of multiple samples.
                # However, to keep the streaming CSV logic (writing line by line), 
                # we will evaluate one by one, although it's slower. 
                # Modern Ragas evaluate() expects a dataset.
                
                # Prepare data for Ragas (new API column names)
                # Truncate context to avoid OOM
                truncated_contexts = [c[:500] for c in contexts] if contexts else [""]
                
                # ground_truth must be a string for RAGAS
                gt = case['ground_truth']
                if isinstance(gt, list):
                    gt = "; ".join(str(x) for x in gt)
                gt = str(gt)
                
                data = {
                    "user_input": [case['question']],
                    "response": [answer],
                    "retrieved_contexts": [truncated_contexts], 
                    "reference": [gt]
                }
                dataset = Dataset.from_dict(data)
                
                # Active Metrics
                active_metrics = []
                
                # Faithfulness: Needs contexts
                if contexts:
                    active_metrics.append(faithfulness)
                
                # Answer Relevancy: Only for specific datasets or if desired
                if case['dataset'] in ['ai2_arc', 'mmlu', 'openbookqa']:
                    active_metrics.append(answer_relevancy)
                
                # Context Recall: Needs Ground Truth
                if case.get('ground_truth_context') and len(case['ground_truth_context']) > 0:
                     gt_ctx = case['ground_truth_context']
                     if isinstance(gt_ctx, list):
                         gt_ctx = "; ".join(str(x) for x in gt_ctx)
                     # Update dataset with GT Context for recall
                     dataset = Dataset.from_dict({
                         **data,
                         "reference": [str(gt_ctx)]
                     })
                     active_metrics.append(context_recall)

                # Execute Ragas (Only if we have metrics to run)
                ragas_scores = {}
                scores = {
                    "faithfulness_score": 0.0,
                    "answer_relevancy_score": 0.0,
                    "context_recall_score": 0.0,
                    "mmlu_accuracy": 0.0
                }
                
                if active_metrics:
                    try:
                        # Initialize LLM/Embeddings for Ragas
                        # Use keep_alive=0 to ensure VRAM is freed after generation
                        langchain_llm = CleanOutputChatOllama(model=MODEL_NAME, keep_alive="0")
                        langchain_embeddings = OllamaEmbeddings(model=MODEL_NAME)
                        
                        results = evaluate(
                            dataset,
                            metrics=active_metrics,
                            llm=langchain_llm,
                            embeddings=langchain_embeddings,
                            raise_exceptions=False 
                        )
                        # Extract scores from EvaluationResult
                        # results.scores is a list of dicts, one per sample
                        if results.scores and len(results.scores) > 0:
                            sample_scores = results.scores[0]
                            for metric_name in ['faithfulness', 'answer_relevancy', 'context_recall']:
                                val = sample_scores.get(metric_name)
                                if val is not None and not math.isnan(val):
                                    ragas_scores[metric_name] = float(val)
                                else:
                                    ragas_scores[metric_name] = float('nan')
                        print(f"  [RAGAS] Scores: {ragas_scores}")
                    except Exception as e:
                        print(f"RAGAS Evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Extract scores (NaN means RAGAS couldn't parse LLM output)
                for metric_key, score_key in [
                    ("faithfulness_score", "faithfulness"),
                    ("answer_relevancy_score", "answer_relevancy"),
                    ("context_recall_score", "context_recall")
                ]:
                    val = ragas_scores.get(score_key)
                    if val is not None and not math.isnan(val):
                        scores[metric_key] = val
                    elif val is not None:
                        scores[metric_key] = "NaN"  # RAGAS ran but LLM output was unparseable

                # Metric: MMLU Accuracy (Strict check)
                if case['dataset'] == 'mmlu':
                    # Check if first character matches ground truth (assumes GT is 'A', 'B', etc.)
                    # Logic: Normalize headers like "Answer: A" -> "A"
                    clean_answer = answer.strip().upper()
                    if clean_answer.startswith(case['ground_truth']):
                         scores["mmlu_accuracy"] = 1.0
                    else:
                         scores["mmlu_accuracy"] = 0.0

                row = {
                    "dataset": case['dataset'],
                    "question": sanitize(case['question']),
                    "ground_truth": sanitize(case['ground_truth']),
                    "model_answer": sanitize(answer),
                    "faithfulness_score": scores['faithfulness_score'],
                    "answer_relevancy_score": scores['answer_relevancy_score'],
                    "context_recall_score": scores['context_recall_score'],
                    "mmlu_accuracy": scores['mmlu_accuracy'],
                    "execution_time": duration
                }
                writer.writerow(row)

    print(f"\nEvaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    run_evaluation()
