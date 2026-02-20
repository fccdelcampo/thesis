import ollama
import json

LLM_MESSAGES = [
    # System Prompt
    {"role": "system",
     "content": """
### ROLE:
You are an expert Educational Assessment Evaluator and Logic Architect. Your task is to filter a dataset of educational test cases (e.g., MMLU, HotpotQA) to identify questions specifically suited for a "Knowledge Graph + Logic Programming" (Prolog-GraphRAG) pipeline.

### EVALUATION CRITERIA:
To receive a "yes" verdict, a test case must satisfy ALL of the following conditions:

1. COMPOSITIONAL FACT-RETRIEVAL:
   The question requires retrieving multiple distinct entities or properties from a knowledge base. It should not be a "common knowledge" question that a model can answer purely through parametric memory without external grounding.

2. PREDICATE LOGIC & RULE APPLICATION:
   The question must require a "logical bridge" to move from retrieved facts to a final answer. 
   - Acceptable: Categorical reasoning (e.g., "Is X a member of Y?"), Transitive properties (e.g., "If A relates to B and B to C..."), or Constraint Satisfaction (e.g., "Which entity satisfies property P but NOT property Q?").
   - Multi-hop Requirement: The answer must be derived by joining at least two separate pieces of information through a logical rule.

3. DISCRETE SYMBOLIC REASONING (NO QUANTITATIVE COMPUTATION):
   Strictly exclude any question that requires:
   - Arithmetic/Calculations: Solving for a numerical variable (even if the steps are logical).
   - Mathematical Proofs: Formal derivations of identities or theorems.
   - Spatial/Visual Reasoning: Reasoning about geometry or physical orientation that cannot be expressed as a logical triple.
   - Constant-Finding: Do not accept questions asking for specific values like "What is the speed of light?" or "Calculate the molar mass."

### REJECTION GUIDELINES (VERDICT: "NO"):
- If the question can be answered by a single Wikipedia snippet: NO.
- If the question contains numbers and asks for a result (e.g., "How many...", "Find the value..."): NO.
- If the reasoning is purely linguistic/semantic rather than logical: NO.

### OUTPUT FORMAT:
Output a single, valid JSON object only. No conversational filler.

### JSON SCHEMA:
{
  "rationale": "Identify the specific facts to be retrieved and the explicit 'if-then' or categorical rules required to bridge them.",
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

import re

def approve(question: str) -> bool:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model='llama3.1:8b-instruct-q4_K_M',
                messages=LLM_MESSAGES + [{"role": "user", "content": question}],
                options={
                    'temperature': 0.7,
                },
                logprobs=True     
            )
            content = response['message']['content']
            
            # Try parsing directly
            try:
                answer = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: Try identifying a code block
                match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if match:
                    answer = json.loads(match.group(1))
                else:
                    # Fallback: Try identifying just a raw JSON start/end even without blocks
                    match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if match:
                         answer = json.loads(match.group(1))
                    else:
                        raise ValueError("No JSON found")

            return answer.get('verdict', '').lower() == 'yes'
            
        except Exception as e:
            # On final retry, log error and fail open or closed?
            # It's better to fail closed (False) so we don't pollute data
            if attempt == max_retries - 1:
                print(f"[test_case_chooser] Error parsing LLM response for '{question[:30]}...': {e}")
                print(f"[test_case_chooser] Raw Content: {content if 'content' in locals() else 'N/A'}")
                return False
            else:
                pass # Retry
    
    return False   