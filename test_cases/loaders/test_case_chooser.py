import ollama
import json

LLM_MESSAGES = [
    # System Prompt
    {"role": "system",
     "content": """
You are an expert Educational Assessment Evaluator. Your task is to filter a dataset of educational test cases (from benchmarks like MMLU) to identify "High-Order Thinking" questions.

### EVALUATION CRITERIA:
To receive a "yes" verdict, a test case must require BOTH of the following:

1. FACT-RETRIEVAL (Knowledge Recall): 
   The question requires the retrieval of specific, established facts from a knowledge base (e.g., historical dates, scientific definitions, or entity properties).

2. LOGICAL REASONING (Rule Application): 
   The question cannot be solved by a single lookup. It requires the application of rules, "if-then" logic, or multi-step synthesis. 
   - Rule Example: "If a planet is closer to the sun than Earth, then its orbital period is shorter than 365 days."
   - Multi-hop Example: "Does [Entity A] share a property with [Entity B]?" (Requires retrieving properties for both and comparing them).

### OUTPUT FORMAT:
Output a single, valid JSON object only. No conversational filler.

### JSON SCHEMA:
{
  "rationale": "A concise explanation of the facts required and the 'if-then' logic or rules needed to reach the conclusion.",
  "verdict": "yes" | "no"
}     
     """},
    
    # Few-Shot Example 1:
    {
        "role": "user", 
        "content": "Which organelle is known as the powerhouse of the cell?"
        },
    {
        "role": "assistant", 
        "content": '{"rationale": "This is a Level 1 (Remembering) question. It requires only the retrieval of a single factual definition (Mitochondria) without any if-then logical processing or rule-based deduction.", "verdict": "no"}'
        },
    
    # Few-Shot Example 2:
    {
        "role": "user", 
        "content": "If Element X has an atomic number of 11 and Element Y is in the same group but one period below, will Element Y be more reactive with water than Element X?"
        },
    {
        "role": "assistant", 
        "content": '{"rationale": "Requires retrieving a fact (Element X is Sodium) and the rule of periodic trends (If an alkali metal is further down a group, then its reactivity increases). The verdict is yes because it necessitates both retrieval and if-then logical application.", "verdict": "yes"}'
        },
    
    # Few-Shot Example 3:
    {
        "role": "user", 
        "content": "Was the treaty that ended the American Revolutionary War signed during the reign of the same British monarch who saw the start of the French Revolution?"
        },
    {
        "role": "assistant", 
        "content": '{"rationale": "Requires retrieving three facts: the Treaty of Paris date (1783), the start of the French Revolution (1789), and the reign of George III. It then requires a logical intersection rule to see if both dates fall within the same temporal range.", "verdict": "yes"}'
        }
]

def approve(question: str) -> bool:
    response = ollama.chat(
            model='llama3.1:8b-instruct-q4_K_M',
            messages=LLM_MESSAGES + [{"role": "user", "content": question}],
            options={
                'temperature': 0.7,
            },
            logprobs=True     
        )
    answer = json.loads(response['message']['content'])
    return answer['verdict'] == 'YES'.lower()   