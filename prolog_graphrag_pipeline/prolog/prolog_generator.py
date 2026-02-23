import janus_swi as janus
import re

def capture_db_and_query(answer_text: str) -> tuple[str, str]:
    import re

    db_match = re.search(r"<database>(.*?)</database>", answer_text, re.DOTALL)
    # if "</query>" not in answer_text:
    #     answer_text += "</query>"
    query_match = re.search(r"<query>(.*?)</query>", answer_text, re.DOTALL)

    try:
        db = db_match.group(1)
    except:
        raise ValueError(f"Database not found in LLM's response \n{answer_text}")
    try:
        query = query_match.group(1)
    except:
        raise ValueError(f"Query not found in LLM's response: \n{answer_text}")

    return {"database":db, "query":query}


def capture_predicate_and_arguments(query: str) -> dict:
    
    predicate_match = re.search(r"(\w+)\s*\(", query)
    try:
        predicate = predicate_match.group(1)
    except:
        raise ValueError("Could not extract predicate name from query")
    
    query = query.replace(".", "").replace("\n", "")
    while query.count("(") > 0 and query.count("(") == query.count(")"):
        ei = query.find("(")
        print(f"prefix: {query[:ei+1]}")
        query = query.replace(query[:ei+1], "").removesuffix(")")
    
    arguments_match = re.search(r"\((.*?)\)", query)
    try:
        arguments = arguments_match.group(1).replace(" ", "").split(",")
        print("Extracted Arguments:", arguments)
    except:
        raise ValueError("Could not extract arguments from query")
    return {"predicate": predicate, "arguments": arguments}

def generate_prolog_generation_prompt(user_question: str, retrieved_context: str, most_recent_error: str) -> str:
    prompt = f"""
Now, process the following:
Context:
{retrieved_context}

User Question: 
{user_question}

---
### RULES:
1. **Strict Prolog Syntax**: 
   - End every fact and rule with a period `.`.
   - Predicates must be lowercase (e.g., `professor(john).` NOT `Professor(John).`).
   - Variables must be uppercase (e.g., `X`, `Person`).
   - Do NOT use hyphens `-` or bullets `*` at the beginning of lines.
   - Do NOT use valid Prolog operators like spaces in invalid ways.
2. **Structure**:
   - Provide the **Database** inside `<database>` tags.
   - Provide the **Query** inside `<query>` tags.
3. **Completeness**:
   - Ensure all predicates used in the query or rules are defined in the database.
   - Do not leave placeholders like `...`.

"""

    if most_recent_error:
        prompt += f"""
### CORRECTION REQUIRED:
Your previous attempt failed with this error:
{most_recent_error}

FIX CODE:
1. If "Operator priority clash", remove any leading hyphens `-` or bullets.
2. If "Singleton variables", ensure variables appear more than once or use `_` for unused ones.
3. If "Unknown procedure", ensure you defined the facts/rules in the database.
"""

    return prompt

from typing import Optional

def generate_prolog_code(user_question: str, retrieved_context: str, most_recent_error: Optional[str]) -> tuple:
    for i in range(5):
        try:
            # Robust Import Logic
            try:
                from .prolog_llms import generate
            except ImportError:
                import sys
                import os
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                try:
                    from prolog_graphrag_pipeline.prolog.prolog_llms import generate
                except ImportError:
                    # Fallback if the package structure is flattened (e.g., tests)
                    from prolog.prolog_llms import generate
            
            prompt = generate_prolog_generation_prompt(user_question, retrieved_context, most_recent_error)
            
            # Explicitly checking signature or arguments is hard here without inspect, but we assume the import worked.
            try:
                answer = generate(prompt=prompt, flag="prolog")
            except TypeError as e:
                # Catch the specific error reported by user if it happens here
                print(f"CRITICAL ERROR calling generate: {e}")
                raise e

            if not answer:
                print(f"Iteration {i} | Error: LLM returned None.")
                continue
            
            answer_text = answer['text_answer']
            
            # CLEANING: Remove markdown bullets if LLM hallucinates them
            lines = answer_text.split('\n')
            cleaned_lines = []
            for line in lines:
                clean = line.strip()
                if clean.startswith("- "):
                    clean = clean[2:]
                elif clean.startswith("* "):
                    clean = clean[2:]
                cleaned_lines.append(clean)
            answer_text = "\n".join(cleaned_lines)

            db_query = capture_db_and_query(answer_text)
            database = db_query["database"]
            query = db_query["query"].strip()
            
            # VALIDATION: Provide fallback for ?- hallucination
            if query.startswith("?-"):
                query = query[2:].strip()

            # VALIDATION: Ensure Query is not just a number
            if re.match(r"^\d+\.?$", query):
                 raise ValueError(f"Invalid Prolog Query: '{query}'. The query must be a Prolog predicate (e.g., 'successor(1, X).'), not a number.")
            
            # VALIDATION: Ensure Query ends with a period
            if not query.endswith('.'):
                query += "."
                
            # VALIDATION: Ensure no unquoted logic symbols exist
            combined_code = database + "\n" + query
            # Remove anything inside single quotes to avoid false positives
            unquoted_code = re.sub(r"'[^']*'", "''", combined_code)
            # Remove Prolog comments (% to end of line) to avoid false positives
            unquoted_code = re.sub(r"%.*$", "", unquoted_code, flags=re.MULTILINE)
            
            # Check for s(CASP) forbidden operators: ;  =:=  =\=  ->  \+  ~  ⊃  ∨  ∧
            # Also check for 'is' used as arithmetic evaluation (word boundary match)
            forbidden_patterns = [
                (r';', "Disjunction ';' is forbidden in s(CASP). Use multiple separate rules with the same head instead of ';'."),
                (r'=:=', "Arithmetic equality '=:=' is forbidden in s(CASP). Use '#=' for constraint arithmetic or '=' for unification."),
                (r'=\\=', "Arithmetic inequality '=\\=' is forbidden in s(CASP). Use '#=\\=' for constraint arithmetic."),
                (r'->', "If-then '->' is forbidden in s(CASP). Use separate rules instead."),
                (r'[~⊃∨∧]', "Unquoted logic symbol found. Wrap logical formulas in single quotes or use 'not'/'-' for negation."),
                (r'\\\+', "Negation '\\+' is forbidden in s(CASP). Use 'not' for negation as failure."),
            ]
            
            validation_failed = False
            for pattern, hint in forbidden_patterns:
                if re.search(pattern, unquoted_code):
                    raise ValueError(f"{hint}")


            print(f"Iteration {i} | Extracted Database:\n{database}\n")
            print(f"Iteration {i} | Extracted Query:\n{query}\n")
            janus.query_once("unload_file(user)")
            janus.consult("user", database)
            
            # Additional Syntax Check: Try to read the query term before executing
            # janus.query_once(f"read_term_from_atom('{query}', Term, [])") 
            
            janus.query_once(query)
            most_recent_error = None
            print(f"Iteration {i} | Prolog code generation success")
            return database, query
        except ValueError as e:
            print(f"Iteration {i} | Validation Error: {e}")
            most_recent_error = f"{e} Ensure the query is a valid Prolog predicate."
            continue
        except janus.PrologError as e:
            print(f"Iteration {i} | Prolog Error: {e}")
            error_str = str(e)
            if "Singleton variables" in error_str:
                hint = "Fix the singleton variables by ensuring each variable appears at least twice in a rule (e.g., replace an unused variable with an underscore '_', or reuse it in the body)."
            elif "Syntax error: Operator expected" in error_str:
                hint = "Wrap any mathematical/logical symbols (=>, ⊃, ∨, ∧, ~, +, -, (, )) entirely in single quotes (like `'(G ∨ H) ⊃ ~I'`) to make them valid Prolog atoms. Do not leave them unquoted."
            else:
                hint = "Ensure strict Prolog syntax. No bullets/hyphens. Predicates must be lowercase."
            most_recent_error = f"Prolog Error: {e}\nHint: {hint}"
            continue
        except Exception as e:
            print(f"Iteration {i} | General Error: {e}")
            if "missing 3 required positional arguments" in str(e):
                raise e # Re-raise this specific error to stop the loop and alert the user
            most_recent_error = f"Error: {e}. Please ensure valid Prolog syntax."
            continue
        
    if most_recent_error:
        raise Exception("Failed to generate valid Prolog code after multiple attempts.")
