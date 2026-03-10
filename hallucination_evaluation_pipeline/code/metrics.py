import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import time
from typing import Tuple

# Add project root to sys.path to allow importing from parallel directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAGAS and Config
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
from datasets import Dataset
from . import ragas_config

from prolog_graphrag_pipeline.llm import generate as llm_generate
from prolog_graphrag_pipeline.config import LLM_SYSTEM_PROMPT

# Define paths
# test_cases/ is two levels up from this file
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results/data")
LOGS_DIR = os.path.join(project_root, "logs")

CORRECTNESS_THRESHOLD = 0.7

FAITHFULNESS_CRITERIA = """
            You are an expert scientific evaluator checking for hallucinations in Physics, Biology, and Chemistry contexts.
Your task is to determine if the claims made in the Generated Answer are entirely logically derived from the provided Context.

CRITICAL PIPELINE INSTRUCTIONS:
The context provided will either be raw text OR a structured logical representation consisting of: Atomic Facts, Conditional Rules, Exceptions, and Logical Gaps.

When evaluating logical derivation, you MUST adhere to the following strict rules:
1. **Structured Logic (If Present):**
   - If 'Exceptions' are listed, ensure the answer does not violate them. If the answer ignores an exception, it is a hallucination.
   - Any conclusion in the answer must strictly follow the 'Conditional Rules'. 
   - If the answer attempts to conclusively resolve a stated 'Logical Gap' without evidence, it is unfaithful.
2. **Raw Text (If Present):**
   - If the context is raw text, verify that the causal links made in the answer are explicitly supported by the text.
3. **Logical Reasoning for Free-Form Questions:**
   - If the context is a structured logical representation, and the answer presents a logical path in its explanation, it should receive a high score.
   - If the context is raw text, and the answer is only an outright claim (even if the claim is correct), it should receive a low score.
4. **Logical Reasoning for Yes/No Questions:**
   - If the context is a structured logical representation, and the answer provides an explanation based on the context after the yes/no answer, it should receive a high score.
   - If the context is raw text, and the answer provides only the final answer without explanation after, it should receive a low score.
5. **Grading Explanations:**
   - The model must have used the provided context to perform logical reasoning and craft an explanation for the answer.
   - The explanation may not be directly from the context, but it should be logically derived from the context.
6. **Noise:**
   - If the context is noisy in terms of unnecessary information or characters, then you should give it a low score.

"""

CONTEXT_RECALL_CRITERIA = """
You are an expert scientific evaluator. 
Your task is to determine if the provided Context contains all the necessary scientific facts, formulas, and logical steps required to arrive at the Ground Truth.

CRITICAL PIPELINE INSTRUCTIONS:
The context will either be raw text OR a 4-part logical structure (Atomic Facts, Conditional Rules, Exceptions, Logical Gaps).

When evaluating if the ground truth is present, you MUST:
1. Break the Ground Truth down into its core scientific components (e.g., specific biological mechanisms, chemical reactions, physical laws).
2. Scan the Context for these exact components.
3. If evaluating the structured logic, check if the Ground Truth can be fully reconstructed using the 'Atomic Facts' and 'Conditional Rules', and other necessary facts/rules that will close any Logical Gaps.
4. If a critical scientific step required by the Ground Truth is listed as a 'Logical Gap', the context recall should be high. But if it is missing entirely from the context, penalize it heavily.
5. The Context must have the necessary facts, rules, and exceptions (if applicable) to answer the question.
6. The Context must be enough for the model to form a helpful and in-depth explanation of the answer.

Do not penalize the context if it contains extra information, unless the information goes against the Ground Truth or contradicts the context.       
    """

ANSWER_RELEVANCY_CRITERIA = """
You are a high school student trying to learn Physics, Chemistry, and Biology.
You are reading an Answer provided by an AI tutor, and your task is to reverse-engineer the exact Question that would have prompted this specific Answer.

CRITICAL INSTRUCTIONS:
1. Read the provided Answer carefully and evaluate how helpful it is to you as a student. 
2. Generate the Question that this Answer seems to be responding to based ONLY on the actual information provided.
3. If the Answer is helpful, clear, and explains the scientific concepts well, generate a highly specific, scientifically accurate question.
4. If the Answer is vague, unhelpful, or relies on conversational filler and multiple-choice formats (e.g., "The answer is A", "Based on my synthesis..."), generate a generic, confused, or off-topic question that reflects the poor quality of the explanation. 
5. Determine if the Answer is noncommittal. Output 1 ONLY if the Answer explicitly states "I don't know", "I am not sure", or completely refuses to answer. If it attempts to answer (even poorly), output 0.     
        """

ANSWER_CORRECTNESS_CRITERIA = """
You are an expert scientific grader evaluating test cases in Physics, Biology, and Chemistry.
Your task is to compare the Generated Answer to the provided Ground Truth label and determine their factual and logical alignment.

CRITICAL OVERRIDE FOR FALSE POSITIVES (FP):
By default, you might classify extra information as a False Positive if it is not explicitly in the Ground Truth. YOU MUST NOT DO THIS for logical reasoning steps. 
- If a statement in the answer represents a 'Conditional Rule', 'Initial Condition', or a 'Logical Step' that accurately explains HOW the conclusion was reached, you MUST classify it as a True Positive (TP), even if the Ground Truth is just a single word like "no" or "false".
- Only classify contradictory or completely irrelevant scientific claims as False Positives (FP).

CRITICAL GRADING RULES:
1. **Strict Scientific Agreement:** If the Generated Answer contradicts the core factual claim, numerical result, or scientific conclusion of the Ground Truth, it must receive a very low score (0.0).
2. **Logical Equivalence:** The answer does not need to match the exact wording of the Ground Truth, provided the underlying scientific logic and final conclusion are identical.
3. If the Generated Answer states that it does not know, or cannot derive the answer from the context, and the Ground Truth is a factual claim, score it as 0.0.
4. If the answer's explanation do not necessarily lead to the core answer, then you should penalize it heavily.
5. If the explanation necessarily leads to the answer, then you should give it a score of 1.0.
    """

FAITHFULNESS_EXTRACTOR_CRITERIA = """
Given a question and an answer, extract the core scientific claims into a list of standalone statements. 

CRITICAL FILTERING RULES:
1. ONLY extract factual claims about Physics, Chemistry, or Biology.
2. DO NOT extract conversational filler, meta-language, or explanations of logical process (e.g., completely ignore phrases like "Based on my synthesis", "This conclusion is reached by", "The system applies the rule", "Therefore", "The Assumption", "The Key Rule").
3. Ensure no pronouns are used. Format the outputs in JSON.
"""

FAITHFULNESS_CRITERIA_2 = """
You are an expert scientific evaluator checking for hallucinations in Physics, Biology, and Chemistry contexts.
Your task is to determine if the claims made in the Generated Answer are logically derived from the provided raw text Context.

CRITICAL OVERRIDE FOR DEDUCTIVE REASONING & META-LANGUAGE:
Standard evaluators often mistakenly penalize answers for showing their logical work or synthesizing facts. YOU MUST NOT DO THIS.
1. **Meta-Language is Faithful:** Phrases like "This conclusion is reached by...", "applying the rule...", "Based on the context...", or "Therefore..." are NOT hallucinations. Do not penalize the answer for explaining its internal reasoning process.
2. **Multi-Hop Deduction is Faithful:** If the context states Fact A (e.g., DNA determines proteins) and Fact B (e.g., spermatogenesis produces cells/sperm), and the answer logically combines them to reach a valid scientific conclusion, this is a FAITHFUL deduction. Do not penalize logical synthesis.
3. **Scientific Synonyms:** Treat scientifically equivalent terms or domain-specific inferences (e.g., equating "sperm" to "cells produced through spermatogenesis") as explicitly supported by the context.

CRITICAL PIPELINE INSTRUCTIONS:
When evaluating logical derivation, you MUST adhere to the following strict rules:
1. **Causal Verification:** Verify that the core scientific claims and logical steps made in the answer are supported by the premises in the text. 
2. **Logical Reasoning for Free-Form Questions:** The answer must present a logical path in its explanation derived from the context. Outright scientific claims without any contextual backing should receive a low score.
3. **True Hallucinations:** Only penalize claims that actively contradict the provided context, or introduce entirely new, outside scientific phenomena that cannot be logically deduced from the raw text.
"""

CONTEXT_RECALL_CRITERIA_2 = """
You are an expert scientific evaluator. 
Your task is to determine if the provided Context contains all the necessary scientific facts, formulas, and logical steps required to arrive at the Ground Truth.

CRITICAL PIPELINE INSTRUCTIONS:
When evaluating if the ground truth is present in the raw text context, you MUST:
1. Break the Ground Truth down into its core scientific components (e.g., specific biological mechanisms, chemical reactions, physical laws).
2. Scan the Context for these exact components.
3. Check if the Ground Truth can be fully reconstructed using the facts and implicit rules present in the text.
4. If a critical scientific step required by the Ground Truth is completely missing from the context, penalize it heavily.
5. The Context must contain enough foundational facts and rules for the model to form a helpful and in-depth explanation of the answer.
6. Do not penalize the context if it contains extra information, unless that information explicitly contradicts the Ground Truth.       
"""

ANSWER_RELEVANCY_CRITERIA_2 = """
You are a high school student.
You ask Physics/Chemistry/Biology questions to an LLM and evaluate its answers on how helpful they are in addressing your question.

When generating potential questions based on the Answer:
1. Focus strictly on the underlying scientific intent. If the question asks about the mechanism of a phenomenon, the answer should explain the physical phenomenon.
2. Ignore conversational filler. Focus entirely on the Physics, Biology, or Chemistry concepts discussed.
3. If the answer is vague or provides a generic response that does not address the specific scientific parameters of the original question, generate questions that reflect that vagueness, which will result in a lower relevancy score.
4. In Yes/No questions, if the answer is the complete opposite of the ground truth, you should give a score of 0.0.
5. In free-form questions, if the answer contains relevant parts of the ground truth answer, and explains it in a much gentler way, you should give a high score.
6. The first part of the answer must match the question type. If the question is free-form or binary (yes/no), the answer should not output a letter choice (e.g. "The answer is A."). If the answer outputs a letter choice, penalize it heavily.
7. The answer must provide a helpful and in-depth explanation.            
"""

ANSWER_CORRECTNESS_CRITERIA_2 = """
You are an expert scientific grader evaluating test cases in Physics, Biology, and Chemistry.
Your task is to compare the Generated Answer to the provided Ground Truth label and determine their factual and logical alignment.

CRITICAL OVERRIDE FOR FALSE POSITIVES (FP):
By default, you might classify extra explanatory information as a False Positive if it is not explicitly written in the Ground Truth. YOU MUST NOT DO THIS for logical reasoning steps. 
- If a statement in the answer represents a logical step, an initial condition, a governing rule, or a premise that accurately explains HOW the conclusion was reached, you MUST classify it as a True Positive (TP), even if the Ground Truth is just a single brief phrase like "no" or "false".
- Only classify contradictory, hallucinated, or completely irrelevant scientific claims as False Positives (FP). Do not penalize the model for showing its logical work.

CRITICAL GRADING RULES:
1. **Strict Scientific Agreement:** If the Generated Answer contradicts the core factual claim, numerical result, or scientific conclusion of the Ground Truth, it must receive a very low score (0.0).
2. **Logical Equivalence:** The answer does not need to match the exact wording of the Ground Truth, provided the underlying scientific logic and final conclusion are identical.
3. If the Generated Answer states that it does not know, or cannot derive the answer from the context, and the Ground Truth is a factual claim, score it as 0.0.
4. If the answer's explanation does not necessarily lead to the core answer, then you should penalize it heavily.
5. If the explanation necessarily leads to the accurate scientific answer, then you should give it a score of 1.0.
"""
# --- CUSTOM PROMPT INJECTIONS ---
    
    # Faithfulness: Ensure the judge respects the logical structure when checking for hallucinations
faithfulness_prompts = faithfulness.get_prompts()
for name, prompt in faithfulness_prompts.items():
    
    # BRANCH 1: Catch and overwrite the Phase 1 Statement Extractor
    if "analyze the complexity" in prompt.instruction.lower() or "extract" in prompt.instruction.lower():
        prompt.instruction = FAITHFULNESS_EXTRACTOR_CRITERIA
        
    # BRANCH 2: Catch and overwrite the Phase 2 NLI Judge
    elif "nli" in name.lower() or "verdict" in name.lower():
        prompt.instruction = FAITHFULNESS_CRITERIA_2
        
    # Apply whichever prompt was just set
    faithfulness.set_prompts(**{name: prompt})

    # Context Recall: Instruct the judge on how to find the ground truth inside the logic
recall_prompts = context_recall.get_prompts()
for name, prompt in recall_prompts.items():
    # if "analyze the complexity" in prompt.instruction.lower():
    #     continue
    if "nli" in name.lower() or "verdict" in name.lower() or "classification" in name.lower() or "recall" in name.lower():
        # Ensure we skip extractors
        if "analyze the complexity" in prompt.instruction.lower() or "extract" in prompt.instruction.lower() or "break down" in prompt.instruction.lower():
             continue
         
        prompt.instruction = (CONTEXT_RECALL_CRITERIA_2)
    context_recall.set_prompts(**{name: prompt})

# Answer Relevancy: Keep the reverse-engineered questions focused on logic
relevancy_prompts = answer_relevancy.get_prompts()
for name, prompt in relevancy_prompts.items():
    # if "analyze the complexity" in prompt.instruction.lower():
    #     continue
    prompt.instruction = (ANSWER_RELEVANCY_CRITERIA_2)
    answer_relevancy.set_prompts(**{name: prompt})
    
    
# Answer Correctness: Keep the reverse-engineered questions focused on logic
correctness_prompts = answer_correctness.get_prompts()
for name, prompt in correctness_prompts.items():
    # if "analyze the complexity" in prompt.instruction.lower():
    #     continue
    
    for name, prompt in answer_correctness.get_prompts().items():
        if "nli" in name.lower() or "classification" in name.lower() or "correctness" in name.lower():
            if "analyze the complexity" in prompt.instruction.lower() or "extract" in prompt.instruction.lower() or "break down" in prompt.instruction.lower():
                continue

            prompt.instruction = (ANSWER_CORRECTNESS_CRITERIA_2)
            answer_correctness.set_prompts(**{name: prompt})


    # --- END CUSTOM PROMPTS ---

def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.
    Specifically targets faithfulness.jsonl (or creates synthetic ones if empty).
    For now, we will use mmlu.jsonl as a proxy if faithfulness.jsonl is empty, 
    or just return an empty list if strict. The plan says use faithfulness.jsonl.
    """
    file_path = os.path.join(TEST_CASES_DIR, "metrics_test_cases.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def run_pipeline_safe(pipeline, pipeline_fn, question: str, **kwargs) -> dict:
    """Safely run a pipeline, returning empty dict on failure. Injects 'elapsed' key."""
    import time
    t0 = time.perf_counter()
    try:
        result = pipeline_fn(question, **kwargs)
    except Exception as e:
        print(f"  Pipeline error in {pipeline} pipeline: {e}")
        result = {"answer": "", "contexts": [], "text_answer": ""}
    result["elapsed"] = time.perf_counter() - t0
    return result


def evaluate_with_ragas(question: str, answer: str, contexts: list[str], ground_truth: str) -> Tuple[float, float, float]:
    """
    Evaluates Faithfulness, Answer Relevancy, Answer Correctness, and Context Recall using RAGAS.
    """
    if not answer or not contexts:
        return 0.0

    # RAGAS expects a dataset
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth] # Faithfulness doesn't strictly need ground truth
    }
    dataset = Dataset.from_dict(data)
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_recall, answer_relevancy, answer_correctness],
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        
        return results['faithfulness'][0], results['context_recall'][0], results['answer_relevancy'][0], results['answer_correctness'][0]
    except Exception as e:
        print(f"RAGAS Evaluation Failed: {e}")
        return 0.0, 0.0, 0.0, 0.0
    
import pandas as pd
from datasets import Dataset
from typing import List

from typing import List
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_relevancy

# Assuming ragas_config is imported

def batch_evaluate_with_ragas(
    questions: List[str], 
    answers: List[str], 
    contexts: List[List[str]], 
    ground_truths: List[str]
) -> pd.DataFrame:
    """
    Evaluates a batch of RAG results and returns a DataFrame with scores per row.
    Includes custom prompt injections to handle structured logical contexts.
    """
    
    # 1. Validation
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("All input lists must have the same length.")
    if len(questions) == 0:
        raise ValueError("Input lists are empty.")

    # 2. Construct the Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # try:
    if True:
        # 3. Run Batch Evaluation
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness], # TODO [faithfulness, context_recall, answer_relevancy, answer_correctness]
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        
        # 4. Convert results to a pandas DataFrame
        return results.to_pandas()
    else:
    # except Exception as e:
        print(f"Batch RAGAS Evaluation Failed: {e}")
        return pd.DataFrame()

# GraphRAG
from graphrag_pipeline.graphrag.graphrag_driver import run_pipeline as run_graphrag

# Prolog GraphRAG
# Import as package from project root
from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag

# Tuned Pipeline (Assuming it exists or we skip)
from tuned_pipeline.main_driver import run_pipeline as run_tuned

def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function.
    """
    question = test_case.get("question")
    ground_truth = test_case.get("answer")
    test_case_id = test_case["test_case_id"]
    if not question:
        return

    print(f"Running Test Case {test_case_id}: {question[:50]}...")

    # 1. Run GraphRAG Pipeline
    try:
        # Expected return: dict with "answer", "context"
        graphrag_result = run_pipeline_safe(pipeline="graphrag", pipeline_fn=run_graphrag, question=question)
        graphrag_answer = graphrag_result.get("answer", "")
        
        print("=========GRAPHRAG OUTPUT:========")
        print(f"{graphrag_answer}")
        print("=================================")
        # Ensure contexts is a list of strings
        # FIX: Extract string content from RetrieverResult
        graphrag_raw_context = graphrag_result.get("contexts", [])
        
        graphrag_processed_ctx = []
        # If it's a list (which RetrieverResult usually is), iterate and extract
        if isinstance(graphrag_raw_context, list):
            for item in graphrag_raw_context:
                # If the item has a 'text' or 'page_content' attribute, use that
                if hasattr(item, 'text'):
                    graphrag_processed_ctx.append(str(item.text))
                elif hasattr(item, 'page_content'):
                    graphrag_processed_ctx.append(str(item.page_content))
                else:
                    graphrag_processed_ctx.append(str(item)) # Fallback to string representation
        else:
            graphrag_processed_ctx = [str(graphrag_raw_context)]
            
        # print("=========GRAPHRAG RETRIEVED CONTEXT:========")
        # print(f"{graphrag_processed_ctx}")
        # print("============================================")
        
        # Evaluate Faithfulness
        faithfulness_gr_score, context_recall_gr_score, answer_relevancy_gr_score, answer_correctness_gr_score = evaluate_with_ragas(question, graphrag_answer, graphrag_processed_ctx, ground_truth)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        faithfulness_gr_score = 0.0
        context_recall_gr_score = 0.0
        answer_relevancy_gr_score = 0.0
        answer_correctness_gr_score = 0.0
    
    # 2. Run Prolog GraphRAG Pipeline
    try:
    # if True:
        # Expected return: dict with "answer", "contexts" (list)
        # run_pipeline(question: str, flag: str, sample_mode: bool = False, include_conclusion: bool = True)
        prolog_result = run_pipeline_safe(pipeline="prolog_graphrag", pipeline_fn=run_prolog_graphrag, question=question, flag="x")
        prolog_answer = prolog_result.get("answer", "")
        
        print("=====PROLOG-GRAPHRAG OUTPUT:=====")
        print(f"{prolog_answer}")
        print("=================================")
        # content is likely a list or string
        # TODO REPLACE LATER IF NECESSARY
        # prolog_graphrag_raw_context = prolog_result.get("condensed_context", [])
        prolog_graphrag_raw_context = prolog_result.get("contexts", [])
        
        prolog_graphrag_processed_ctx = []
        # If it's a list (which RetrieverResult usually is), iterate and extract
        if isinstance(prolog_graphrag_raw_context, list):
            for item in prolog_graphrag_raw_context:
                # If the item has a 'text' or 'page_content' attribute, use that
                if hasattr(item, 'text'):
                    prolog_graphrag_processed_ctx.append(str(item.text))
                elif hasattr(item, 'page_content'):
                    prolog_graphrag_processed_ctx.append(str(item.page_content))
                else:
                    prolog_graphrag_processed_ctx.append(str(item)) # Fallback to string representation
        else:
            prolog_graphrag_processed_ctx = [str(prolog_graphrag_raw_context)]
            
        # print("=========PROLOG-GRAPHRAG RETRIEVED CONTEXT:========")
        # print(f"{prolog_graphrag_processed_ctx}")
        # print("===================================================")
        
        # Evaluate Faithfulness
        faithfulness_pgr_score, context_recall_pgr_score, answer_relevancy_pgr_score, answer_correctness_pgr_score = evaluate_with_ragas(question, prolog_answer, prolog_graphrag_processed_ctx, ground_truth)
    except Exception as e:
    # else:
        print(f"Prolog GraphRAG Failed: {e}")
        faithfulness_pgr_score = 0.0
        context_recall_pgr_score = 0.0
        answer_relevancy_pgr_score = 0.0
        answer_correctness_pgr_score = 0.0
    
    # Log Results
    log_results(test_case_id, faithfulness_gr_score, faithfulness_pgr_score, "faithfulness")
    log_results(test_case_id, context_recall_gr_score, context_recall_pgr_score, "context_recall")
    log_results(test_case_id, answer_relevancy_gr_score, answer_relevancy_pgr_score, "answer_relevancy")
    log_results(test_case_id, answer_correctness_gr_score, answer_correctness_pgr_score, "answer_correctness")
    log_test_case_run(
        test_case_id=test_case_id, 
        graphrag_answer=graphrag_answer, 
        graphrag_contexts=graphrag_processed_ctx,
        graphrag_faithfulness_score=faithfulness_gr_score,
        graphrag_answer_relevancy_score=answer_relevancy_gr_score,
        graphrag_answer_correctness_score=answer_correctness_gr_score,
        graphrag_context_recall=context_recall_gr_score,
        prolog_graphrag_answer=prolog_answer,
        prolog_graphrag_contexts=prolog_graphrag_processed_ctx,
        prolog_graphrag_faithfulness_score=faithfulness_pgr_score,
        prolog_graphrag_answer_relevancy_score=answer_relevancy_pgr_score,
        prolog_graphrag_answer_correctness_score=answer_correctness_pgr_score,
        prolog_graphrag_context_recall=context_recall_pgr_score
        )
    
    
from typing import List, Dict
import pandas as pd

def run_test_case_batch(test_cases: List[Dict], start_index: int = 0, end_index: int = 300):
    """
    Runs a batch of test cases for both GraphRAG and Prolog GraphRAG,
    then evaluates them in bulk using RAGAS.
    """
    # Storage for pipeline outputs
    gr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    pgr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    test_case_ids = []

    # 1. Execute all pipeline runs first
    for i, test_case in enumerate(test_cases[start_index: end_index]):
        question = test_case.get("question")
        ground_truth = test_case.get("answer", "")
        print(f"Processing Pipeline Runs for Test Case {i}: {question}...")
        print("=====GROUND TRUTH ANSWER:=====")
        print(f"{ground_truth}")
        print("=================================")
        if not question:
            continue
        
        test_case_ids.append(i)

        # --- GraphRAG Run ---
        try:
            graphrag_result = run_graphrag(question)
            graphrag_answer = graphrag_result.get("answer", "")
            
            print("========GRAPHRAG OUTPUT:=========")
            print(f"{graphrag_answer}")
            print("=================================")
            
            # FIX: Extract string content from RetrieverResult
            graphrag_raw_ctx = graphrag_result.get("contexts", [])
            
            graphrag_processed_ctx = []
            # If it's a list (which RetrieverResult usually is), iterate and extract
            if isinstance(graphrag_raw_ctx, list):
                for item in graphrag_raw_ctx:
                    # If the item has a 'text' or 'page_content' attribute, use that
                    if hasattr(item, 'text'):
                        graphrag_processed_ctx.append(str(item.text))
                    elif hasattr(item, 'page_content'):
                        graphrag_processed_ctx.append(str(item.page_content))
                    else:
                        graphrag_processed_ctx.append(str(item)) # Fallback to string representation
            else:
                graphrag_processed_ctx = [str(graphrag_raw_ctx)]

            gr_data["questions"].append(question)
            gr_data["answers"].append(graphrag_answer)
            gr_data["contexts"].append(graphrag_processed_ctx) # Now a clean list of strings
            gr_data["ground_truths"].append(ground_truth)
        except Exception as e:
            print(f"GraphRAG Execution failed for case {i}: {e}")

        # --- Prolog GraphRAG Run ---
        # try:
        if True:
            prolog_graphrag_result = run_pipeline_safe(pipeline="prolog_graphrag", pipeline_fn=run_prolog_graphrag, question=question, flag="x")
            prolog_graphrag_answer = prolog_graphrag_result.get("answer", "")
            print("=====PROLOG-GRAPHRAG OUTPUT:=====")
            print(f"{prolog_graphrag_answer}")
            print("=================================")
            # FIX: Extract string content from RetrieverResult
            # TODO REPLACE LATER IF NECESSARY
            # prolog_graphrag_raw = prolog_graphrag_result.get("condensed_context", [])
            prolog_graphrag_raw = prolog_graphrag_result.get("contexts", [])
            
            prolog_graphrag_processed_ctx = []
            # If it's a list (which RetrieverResult usually is), iterate and extract
            if isinstance(prolog_graphrag_raw, list):
                for item in prolog_graphrag_raw:
                    # If the item has a 'text' or 'page_content' attribute, use that
                    if hasattr(item, 'text'):
                        prolog_graphrag_processed_ctx.append(str(item.text))
                    elif hasattr(item, 'page_content'):
                        prolog_graphrag_processed_ctx.append(str(item.page_content))
                    else:
                        prolog_graphrag_processed_ctx.append(str(item)) # Fallback to string representation
            else:
                prolog_graphrag_processed_ctx = [str(prolog_graphrag_raw)]
            
            pgr_data["questions"].append(question)
            pgr_data["answers"].append(prolog_graphrag_answer)
            pgr_data["contexts"].append(prolog_graphrag_processed_ctx)
            pgr_data["ground_truths"].append(ground_truth)
        # except Exception as e:
        else:
            print(f"Prolog GraphRAG Execution failed for case {i}: {e}")

    # 2. Perform Batch Evaluation
    print("\nStarting Batch RAGAS Evaluation...")
    
    df_gr = batch_evaluate_with_ragas(
        gr_data["questions"], gr_data["answers"], gr_data["contexts"], gr_data["ground_truths"]
    )
    
    df_pgr = batch_evaluate_with_ragas(
        pgr_data["questions"], pgr_data["answers"], pgr_data["contexts"], pgr_data["ground_truths"]
    )

    # 3. Log Results per test case
    # Assuming the order in the resulting DataFrame matches the input order
    for idx, t_id in enumerate(test_case_ids):
        # Log GraphRAG vs Prolog GraphRAG for each metric
        for metric in ["faithfulness", "context_recall", "answer_relevancy", "answer_correctness"]:
            score_gr = df_gr.iloc[idx][metric] if not df_gr.empty else 0.0
            score_pgr = df_pgr.iloc[idx][metric] if not df_pgr.empty else 0.0
            log_results(t_id, score_gr, score_pgr, metric)
            

    return df_gr, df_pgr

def record_test_case_batch(test_cases: List[Dict], start_index: int | None = 0, end_index: int | None = 300):
    """
    Runs a batch of test cases for both GraphRAG and Prolog GraphRAG,
    then evaluates them in bulk using RAGAS.
    """
    # Storage for pipeline outputs
    gr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    pgr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": [], "fallback": []}
    test_case_ids = []

    # 1. Execute all pipeline runs first
    for i, test_case in enumerate(test_cases[start_index:end_index]):
        question = test_case.get("question")
        ground_truth = test_case.get("answer", "")
        print("=" * 80)
        print(f"Processing Pipeline Runs for Test Case {start_index + i}: {question}...")
        print("=====GROUND TRUTH ANSWER:=====")
        print(f"{ground_truth}")
        print("=================================")
        if not question:
            continue
        
        test_case_ids.append(i)


        # --- GraphRAG Run ---
        try:
            graphrag_result = run_graphrag(question)
            graphrag_answer = graphrag_result.get("answer", "")
            print("========GRAPHRAG OUTPUT:=======")
            print(f"{graphrag_answer}")
            print("=================================")
            
            # FIX: Extract string content from RetrieverResult
            graphrag_raw_ctx = graphrag_result.get("contexts", [])
            
            graphrag_processed_ctx = []
            # If it's a list (which RetrieverResult usually is), iterate and extract
            if isinstance(graphrag_raw_ctx, list):
                for item in graphrag_raw_ctx:
                    # If the item has a 'text' or 'page_content' attribute, use that
                    if hasattr(item, 'text'):
                        graphrag_processed_ctx.append(str(item.text))
                    elif hasattr(item, 'page_content'):
                        graphrag_processed_ctx.append(str(item.page_content))
                    else:
                        graphrag_processed_ctx.append(str(item)) # Fallback to string representation
            else:
                graphrag_processed_ctx = [str(graphrag_raw_ctx)]

            gr_data["questions"].append(question)
            gr_data["answers"].append(graphrag_answer)
            gr_data["contexts"].append(graphrag_processed_ctx) # Now a clean list of strings
            gr_data["ground_truths"].append(ground_truth)
        except Exception as e:
            print(f"GraphRAG Execution failed for case {i}: {e}")

        # --- Prolog GraphRAG Run ---
        # try:
        if True:
            prolog_graphrag_result = run_pipeline_safe(pipeline="prolog_graphrag", pipeline_fn=run_prolog_graphrag, question=question, flag="x")
            prolog_graphrag_answer = prolog_graphrag_result.get("answer", "")
            fallback = prolog_graphrag_result.get("fallback", "")
            print("=====PROLOG-GRAPHRAG OUTPUT:=====")
            print(f"{prolog_graphrag_answer}")
            print("=================================")
            # FIX: Extract string content from RetrieverResult
            # TODO REPLACE LATER IF NECESSARY
            # prolog_graphrag_raw = prolog_graphrag_result.get("condensed_context", [])
            prolog_graphrag_raw = prolog_graphrag_result.get("contexts", [])
            
            prolog_graphrag_processed_ctx = []
            # If it's a list (which RetrieverResult usually is), iterate and extract
            if isinstance(prolog_graphrag_raw, list):
                for item in prolog_graphrag_raw:
                    # If the item has a 'text' or 'page_content' attribute, use that
                    if hasattr(item, 'text'):
                        prolog_graphrag_processed_ctx.append(str(item.text))
                    elif hasattr(item, 'page_content'):
                        prolog_graphrag_processed_ctx.append(str(item.page_content))
                    else:
                        prolog_graphrag_processed_ctx.append(str(item)) # Fallback to string representation
            else:
                prolog_graphrag_processed_ctx = [str(prolog_graphrag_raw)]
            
            pgr_data["questions"].append(question)
            pgr_data["answers"].append(prolog_graphrag_answer)
            pgr_data["contexts"].append(prolog_graphrag_processed_ctx)
            pgr_data["fallback"].append(fallback)
            pgr_data["ground_truths"].append(ground_truth)
        # except Exception as e:
        else:
            print(f"Prolog GraphRAG Execution failed for case {i}: {e}")
            
    file_path = os.path.join(LOGS_DIR, "metrics.jsonl")
    with open(file_path, 'a', encoding='utf-8') as f:
        for i, test_case in enumerate(test_cases[start_index:end_index]):
            row = {
                "test_case_id": test_case["test_case_id"],
                "question": test_case["question"],
                "graphrag_answer": gr_data["answers"][i],
                "graphrag_contexts": gr_data["contexts"][i],
                "prolog_graphrag_answer": pgr_data["answers"][i],
                "prolog_graphrag_contexts": pgr_data["contexts"][i],
                "fallback": pgr_data["fallback"][i],
                "ground_truth": test_case["answer"]
            }
            f.write(json.dumps(row) + "\n")
            
def batch_evaluate(read_all: bool = True, start_index: int | None = 0, end_index: int | None = 300):
    test_case_ids = []
    questions = []
    graphrag_contexts = []
    graphrag_answers = []
    prolog_graphrag_contexts = []
    prolog_graphrag_answers = []
    fallback = []
    ground_truths = []
    
    if start_index >= end_index:
        raise ValueError("start_index is greater than end index.")
    
    with open(os.path.join(LOGS_DIR, "metrics.jsonl"), "r") as f:
        lines = f.readlines() # Reads all lines into a list in memory
        lines_to_read = lines if read_all else lines[start_index:end_index]
        print(f"Evaluating {len(lines_to_read)} test cases...")
    
        for line in lines_to_read:
            row = json.loads(line)
            test_case_ids.append(row["test_case_id"])
            questions.append(row["question"])
            graphrag_contexts.append(row["graphrag_contexts"])
            graphrag_answers.append(row["graphrag_answer"])
            prolog_graphrag_contexts.append(row["prolog_graphrag_contexts"])
            prolog_graphrag_answers.append(row["prolog_graphrag_answer"])
            fallback.append(row["fallback"])
            ground_truths.append(row["ground_truth"])
        
    # 2. Perform Batch Evaluation
    print("\nStarting Batch RAGAS Evaluation...")
    
    df_gr = batch_evaluate_with_ragas(
        questions, graphrag_answers, graphrag_contexts, ground_truths
    )
    
    # df_pgr = []
    df_pgr = batch_evaluate_with_ragas(
        questions, prolog_graphrag_answers, prolog_graphrag_contexts, ground_truths
    )
    
    for idx, t_id in enumerate(test_case_ids):
        # Log GraphRAG vs Prolog GraphRAG for each metric
        for metric in ["faithfulness"]: # TODO ["faithfulness", "context_recall", "answer_relevancy", "answer_correctness"]
            score_gr = df_gr.iloc[idx][metric] if not df_gr.empty else 0.0
            score_pgr = df_pgr.iloc[idx][metric] if not df_pgr.empty else 0.0
            Fallback = fallback[idx]
            log_results(t_id, score_gr, score_pgr, Fallback, metric)
            

    return df_gr, df_pgr

def log_results(test_case_id: int, graphrag_score: float, prolog_graphrag_score: float, fallback: str, metric: str) -> None:
    if metric == "faithfulness":
        file_path = os.path.join(RESULTS_DIR, "faithfulness_scores.csv")
    elif metric == "context_recall":
        file_path = os.path.join(RESULTS_DIR, "context_recall_scores.csv")
    elif metric == "answer_relevancy":
        file_path = os.path.join(RESULTS_DIR, "answer_relevancy_scores.csv")
    elif metric == "answer_correctness":
        file_path = os.path.join(RESULTS_DIR, "answer_correctness_scores.csv")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    new_row = pd.DataFrame({
        "test_case_id": [test_case_id],
        "graphrag_score": [graphrag_score],
        "graphrag_label": [0 if graphrag_score >= CORRECTNESS_THRESHOLD else 1],
        "prolog_graphrag_score": [prolog_graphrag_score],
        "prolog_graphrag_label": [0 if prolog_graphrag_score >= CORRECTNESS_THRESHOLD else 1],
        "fallback": [fallback]
    })
    
    header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    new_row.to_csv(file_path, mode='a', header=header, index=False)
    
    
def log_test_case_run(
    test_case_id: int, 
    graphrag_answer: str, 
    graphrag_contexts: str | list[str], 
    graphrag_faithfulness_score: float, 
    graphrag_answer_relevancy_score: float, 
    graphrag_answer_correctness_score: float, 
    graphrag_context_recall: float, 
    prolog_graphrag_answer: str, 
    prolog_graphrag_contexts: str | list[str], 
    prolog_graphrag_faithfulness_score: float, 
    prolog_graphrag_answer_relevancy_score: float, 
    prolog_graphrag_answer_correctness_score: float, 
    prolog_graphrag_context_recall: float
) -> None:
    graphrag_file_path = os.path.join(LOGS_DIR, "graphrag_logs/metrics.csv")
    prolog_graphrag_file_path = os.path.join(LOGS_DIR, "prolog_graphrag_logs/metrics.jsonl")

    graphrag_entry = {
        "test_case_id": test_case_id,
        "graphrag_answer": graphrag_answer,
        "graphrag_contexts": graphrag_contexts,
        "graphrag_faithfulness_score": graphrag_faithfulness_score,
        "graphrag_answer_relevancy_score": graphrag_answer_relevancy_score,
        "graphrag_answer_correctness_score": graphrag_answer_correctness_score,
        "graphrag_context_recall": graphrag_context_recall,
    }
    
    prolog_graphrag_entry = {
        "test_case_id": test_case_id,
        "prolog_graphrag_answer": prolog_graphrag_answer,
        "prolog_graphrag_contexts": prolog_graphrag_contexts,
        "prolog_graphrag_faithfulness_score": prolog_graphrag_faithfulness_score,
        "prolog_graphrag_answer_relevancy_score": prolog_graphrag_answer_relevancy_score,
        "prolog_graphrag_answer_correctness_score": prolog_graphrag_answer_correctness_score,
        "prolog_graphrag_context_recall": prolog_graphrag_context_recall,
    }
    
    with open(graphrag_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(graphrag_entry) + "\n")
        
    with open(prolog_graphrag_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(prolog_graphrag_entry) + "\n")

def main():
    start_index, end_index = 0, 100
    test_cases = load_test_cases()
    # run_test_case_batch(test_cases)
    print(f"Loaded {len(test_cases)} test cases for Faithfulness, Answer Relevancy, Answer Correctness, and Context Recall.")
    record_test_case_batch(test_cases, start_index=start_index, end_index=end_index)
    batch_evaluate(read_all=True, start_index=start_index, end_index=end_index)
    # for i, test_case in enumerate(tqdm(test_cases)):
    #     run_test_case(test_case, i)
    
def collate_responses():
    start_index, end_index = 100, 101
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases for Faithfulness, Answer Relevancy, Answer Correctness, and Context Recall.")
    record_test_case_batch(test_cases, start_index=start_index, end_index=end_index)
    
def start_evaluations():
    start_index = 0
    end_index = 300
    batch_evaluate(read_all=True, start_index=start_index, end_index=end_index)


if __name__ == "__main__":
    # main()
    pass