import ollama
import json

LLM_MESSAGES = [
    # System Prompt
    {"role": "system",
     "content": """
### ROLE:
You are an expert Logical Architect and Knowledge Engineer. Your task is to evaluate educational test cases (MMLU, HotpotQA) to determine if they are suitable for a Symbolic Reasoning pipeline (Prolog-GraphRAG). 

### CORE OBJECTIVE:
Identify "Symbolic-Friendly" questions that require retrieving facts AND applying at least two discrete "if-then" logical rules to reach a conclusion.

### INCLUSION CRITERIA (VERDICT: "YES"):
To be accepted, the question MUST satisfy these three pillars:

1. MULTI-HOP FACT RETRIEVAL:
   - The question cannot be answered by a single entity lookup. 
   - It must require connecting at least two distinct entities or concepts via a shared property or relation (e.g., A → B → C).

2. MULTI-STEP RULE APPLICATION (IF-THEN):
   - The reasoning must require at least TWO logical rules/predicates.
   - Example Rules: Transitivity (If A > B and B > C, then A > C), Class Inheritance (If X is a Y, and all Ys have Z...), or Negation (If P is true, then Q cannot be R).
   - The "rationale" field must explicitly list these rules.

3. PURELY SYMBOLIC / NON-COMPUTATIONAL:
   - The question must be "educational" (Science, Law, Medicine, Philosophy).
   - It must deal with discrete categories and relationships, NOT continuous values or math.
   
4. DIFFICULTY:
   - The question should be difficult for an LLM to answer correctly without symbolic reasoning.
   - It should not be a simple fact recall or a straightforward multi-hop that doesn't require complex rule application.

### EXCLUSION CRITERIA (VERDICT: "NO"):
- CALCULATIONS: Any question requiring arithmetic, statistics, or numerical solving (even simple addition).
- PROOFS: Formal mathematical derivations or algebraic manipulations.
- SINGLE-STEP: "What is the capital of X?" or "Who wrote Y?"
- SEMANTIC ONLY: Questions about word definitions or grammar that don't involve logical "world rules."

### OUTPUT FORMAT:
Return ONLY a JSON object. No preamble.

### JSON SCHEMA:
{
  "rationale": "1. Facts to retrieve: [List facts]. 2. Rule 1: [If-Then statement]. 3. Rule 2: [If-Then statement]. 4. Connection: [How they lead to the answer].",
  "verdict": "yes" | "no"
}  
     """},
    
    # Few-Shot Example 1:
    {
        "role": "user", 
        "content": """
In a specific ecosystem, the population of wolves (Predator A) keeps the deer population (Herbivore B) in check. The deer primarily feed on young willow trees (Producer C). If a disease significantly reduces the wolf population, what is the most likely secondary effect on the ecosystem?
(A) The deer population will decrease.
(B) The willow tree population will decrease.
(C) The soil nutrients will increase immediately.
(D) The deer will become carnivores.
        """
        },
    {
        "role": "assistant", 
        "content": '{"rationale": "1. Facts: Wolves prey on Deer; Deer eat Willows. 2. Rule 1 (Inverse Correlation): If predator population decreases, prey population increases. 3. Rule 2 (Consumption): If herbivore population increases, producer population decreases. 4. Connection: Chains the reduction of wolves to the increase of deer, which leads to the depletion of willows (B).", "verdict": "yes"}'
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
        },
    
    # Few-Shot Example 4:
    {
        "role": "user", 
        "content": "True or False: The Battle of Hastings took place in the year 1066."
        },
    {
        "role": "assistant",
        "content": '{"rationale": "This is a single-hop historical fact retrieval. It does not require a logical \'if-then\' bridge or multi-hop reasoning across distinct entities; it is a direct attribute lookup for a single event.", "verdict": "no"}'
        },
    
    # Few-Shot Example 5:
    {
        "role": "user", 
        "content": "Under a specific jurisdiction, a contract is 'voidable' if it was signed under duress. Person A signed a deed while being physically threatened by Person B. True or False: Person A has the legal right to rescind this deed."
        },
    {
        "role": "assistant",
        "content": """{"rationale": "1. Facts: The deed is a contract; Person A was physically threatened; Physical threats constitute 'duress'. 2. Rule 1: If an action is performed under duress, the resulting contract is voidable. 3. Rule 2: If a contract is voidable, the signer has the right to rescind. 4. Connection: Maps the specific instance (threat) to the legal category (duress) to derive the right to rescind.", "verdict": "yes"}"""
        },
    
    # Few-Shot Example 6:
    {
        "role": "user", 
        "content": "A solution has a pH of 4.0. What is the hydrogen ion concentration [H+] in moles per liter? (A) 10^-4 (B) 10^4 (C) 4.0 (D) 0.0004"
        },
    {
        "role": "assistant",
        "content": '{"rationale": "This question requires a numerical calculation based on the formula pH = -log[H+]. It is a quantitative mathematical operation rather than a symbolic logical derivation between predicates.", "verdict": "no"}'
        },
    
    # Few-Shot Example 7:
    {
        "role": "user", 
        "content": """
All cephalopods have tentacles. An octopus is a type of cephalopod. Blue-ringed octopuses are a subset of the octopus family. What can be logically inferred?
(A) Only some octopuses have tentacles.
(B) Blue-ringed octopuses do not have tentacles.
(C) Blue-ringed octopuses have tentacles.
(D) Cephalopods are a subset of octopuses.
"""
        },
    {
        "role": "assistant",
        "content": '{"rationale": "1. Facts: Blue-ringed Octopus is-a Octopus; Octopus is-a Cephalopod; Cephalopod has-property Tentacles. 2. Rule 1 (Transitive Inheritance): If A is-a B and B is-a C, then A is-a C. 3. Rule 2 (Property Inheritance): If X is-a Y, then X inherits all universal properties of Y. 4. Connection: Derives that Blue-ringed octopuses have tentacles (C) via a multi-level class hierarchy.", "verdict": "yes"}'
        },
]

import re

def approve(question: str) -> bool:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model='llama3.2:3b-instruct-q8_0',
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
            verdict = answer.get('verdict', '').lower() == 'yes'
            
            if verdict:
                print(f"[test_case_chooser] APPROVED. (Rationale: {answer.get('rationale', 'N/A')})")
            else:
                print(f"[test_case_chooser] ~NOT APPROVED~. (Rationale: {answer.get('rationale', 'N/A')})")
            return verdict
            
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