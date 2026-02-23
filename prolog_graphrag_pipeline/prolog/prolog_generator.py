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

def generate_prolog_generation_prompt(user_question: str, retrieved_context: str, error_history: list) -> str:
    prompt = f"""Now, process the following:
Context:
{retrieved_context}

User Question:
{user_question}
"""

    if error_history:
        numbered = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(error_history))
        prompt += f"""
### FIX REQUIRED
The following errors occurred across prior attempts. Fix ALL of them before responding.

{numbered}

Locate every offending line in your previous database/query and fix it.
Rewrite ONLY what is broken. Output the corrected code in `<database>` / `<query>` tags.
"""

    return prompt

from typing import Optional


def _ask_if_close_to_fixing(generate_fn, most_recent_error: str) -> bool:
    """
    Ask the LLM whether the last Prolog error looks easily fixable.
    Returns True if the LLM thinks so (→ grant 5 more attempts),
    False otherwise (→ raise immediately).
    """
    close_check_prompt = f"""You are evaluating a Prolog code generation failure.

The last error encountered was:
{most_recent_error}

Is this error likely fixable with one or two small adjustments to the Prolog code?
Answer with ONLY one word: YES or NO.
"""
    try:
        result = generate_fn(prompt=close_check_prompt, flag="q")
        if not result:
            return False
        answer_text = result.get("text_answer", "").strip().upper()
        print(f"[Prolog retry] Close-check answer: {repr(answer_text)}")
        return answer_text.startswith("YES")
    except Exception as e:
        print(f"[Prolog retry] Close-check LLM call failed: {e}. Skipping extension.")
        return False


def _run_prolog_attempt(generate_fn, i: int, user_question: str, retrieved_context: str, error_history: list):
    """Single attempt of Prolog code generation + validation. Returns (database, query) on success."""
    prompt = generate_prolog_generation_prompt(user_question, retrieved_context, error_history)

    try:
        answer = generate_fn(prompt=prompt, flag="prolog")
    except TypeError as e:
        print(f"CRITICAL ERROR calling generate: {e}")
        raise e

    if not answer:
        raise ValueError(f"Iteration {i} | LLM returned None.")

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
    unquoted_code = re.sub(r"'[^']*'", "''", combined_code)
    unquoted_code = re.sub(r"%.*$", "", unquoted_code, flags=re.MULTILINE)

    forbidden_patterns = [
        (r';', "Disjunction ';' is forbidden in s(CASP). Use multiple separate rules with the same head instead of ';'."),
        (r'=:=', "Arithmetic equality '=:=' is forbidden in s(CASP). Use '#=' for constraint arithmetic or '=' for unification."),
        (r'=\\=', "Arithmetic inequality '=\\=' is forbidden in s(CASP). Use '#=\\=' for constraint arithmetic."),
        (r'->', "If-then '->' is forbidden in s(CASP). Use separate rules instead."),
        (r'[~⊃∨∧]', "Unquoted logic symbol found. Wrap logical formulas in single quotes or use 'not'/'-' for negation."),
        (r'\\\\\\+', "Negation '\\+' is forbidden in s(CASP). Use 'not' for negation as failure."),
    ]

    for pattern, hint in forbidden_patterns:
        if re.search(pattern, unquoted_code):
            raise ValueError(f"{hint}")

    print(f"Iteration {i} | Extracted Database:\n{database}\n")
    print(f"Iteration {i} | Extracted Query:\n{query}\n")
    janus.query_once("unload_file(user)")
    janus.consult("user", database)
    janus.query_once(query)

    print(f"Iteration {i} | Prolog code generation success")
    return database, query


def generate_prolog_code(user_question: str, retrieved_context: str, most_recent_error: Optional[str]) -> tuple:
    # ── Resolve the generate function once ───────────────────────────────────
    try:
        from .prolog_llms import generate as generate_fn
    except ImportError:
        import sys, os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from prolog_graphrag_pipeline.prolog.prolog_llms import generate as generate_fn
        except ImportError:
            from prolog.prolog_llms import generate as generate_fn

    # ── Phase 1: Normal attempts (5 tries) ───────────────────────────────────
    NORMAL_ATTEMPTS = 5
    EXTENSION_ATTEMPTS = 5
    error_history: list = []  # accumulates every unique error across all iterations

    def _record_error(msg: str):
        """Append error to history, deduplicating consecutive identical messages."""
        if not error_history or error_history[-1] != msg:
            error_history.append(msg)

    for i in range(NORMAL_ATTEMPTS):
        try:
            return _run_prolog_attempt(generate_fn, i, user_question, retrieved_context, error_history)
        except ValueError as e:
            err_str = str(e)
            print(f"Iteration {i} | Validation Error: {err_str}")
            if "Disjunction" in err_str or "';'" in err_str:
                hint = ("Forbidden: ';' (disjunction) is not allowed in s(CASP). "
                        "Replace it with multiple rules that share the same head. "
                        "Example — WRONG: `answer(X) :- option_a(X) ; option_b(X).` "
                        "RIGHT: `answer(X) :- option_a(X).` and `answer(X) :- option_b(X).`")
                _record_error(f"{err_str} {hint}")
            else:
                _record_error(f"{err_str} Ensure the query is a valid Prolog predicate.")
        except janus.PrologError as e:
            print(f"Iteration {i} | Prolog Error: {e}")
            error_str = str(e)
            if "Singleton variables" in error_str:
                hint = "Fix the singleton variables by ensuring each variable appears at least twice in a rule (e.g., replace an unused variable with an underscore '_', or reuse it in the body)."
            elif "Syntax error: Operator expected" in error_str:
                hint = "Wrap any mathematical/logical symbols (=>, ⊃, ∨, ∧, ~, +, -, (, )) entirely in single quotes (like `'(G ∨ H) ⊃ ~I'`) to make them valid Prolog atoms. Do not leave them unquoted."
            elif "not sufficiently instantiated" in error_str:
                hint = ("You used an UPPERCASE variable as a constant in a database fact (e.g., `option(A, cell_wall).` where A is unbound). "
                        "Use LOWERCASE atoms for constants in facts (e.g., `option(a, cell_wall).`). "
                        "Variables (uppercase) should only appear in rule heads/bodies where they are instantiated by matching facts.")
            elif "Unknown procedure" in error_str:
                hint = ("Arity mismatch: the predicate called in the query has a different number of arguments than what is defined in the database. "
                        "Check that every predicate call in the query uses the exact same number of arguments as its definition (e.g., if DB has "
                        "`effect(cause, result)` then the query must use `effect(Cause, Result)`, NOT `effect(Cause)`). "
                        "Also ensure every predicate used in the query is actually defined in the database.")
            else:
                hint = "Ensure strict Prolog syntax. No bullets/hyphens. Predicates must be lowercase."
            _record_error(f"Prolog Error: {e}\nHint: {hint}")
        except Exception as e:
            print(f"Iteration {i} | General Error: {e}")
            if "missing 3 required positional arguments" in str(e):
                raise e  # Fatal — stop immediately
            _record_error(f"Error: {e}. Please ensure valid Prolog syntax.")

    # ── Phase 2: Close-check — should we try 5 more times? ───────────────────
    most_recent_error = error_history[-1] if error_history else None
    print(f"[Prolog retry] Normal attempts exhausted. Asking LLM if error is close to fixable...")
    if most_recent_error and _ask_if_close_to_fixing(generate_fn, most_recent_error):
        print(f"[Prolog retry] LLM says YES — granting {EXTENSION_ATTEMPTS} extension attempts.")
        for i in range(NORMAL_ATTEMPTS, NORMAL_ATTEMPTS + EXTENSION_ATTEMPTS):
            try:
                return _run_prolog_attempt(generate_fn, i, user_question, retrieved_context, error_history)
            except ValueError as e:
                print(f"Extension {i} | Validation Error: {e}")
                _record_error(f"{e} Ensure the query is a valid Prolog predicate.")
            except janus.PrologError as e:
                print(f"Extension {i} | Prolog Error: {e}")
                error_str = str(e)
                if "Singleton variables" in error_str:
                    hint = "Fix the singleton variables by ensuring each variable appears at least twice in a rule."
                elif "Syntax error: Operator expected" in error_str:
                    hint = "Wrap mathematical/logical symbols in single quotes."
                elif "not sufficiently instantiated" in error_str:
                    hint = ("You used an UPPERCASE variable as a constant in a database fact (e.g., `option(A, cell_wall).`). "
                            "Use LOWERCASE atoms instead (e.g., `option(a, cell_wall).`).")
                elif "Unknown procedure" in error_str:
                    hint = ("Arity mismatch: ensure the predicate in the query uses the same number of arguments as its definition in the database. "
                            "Also ensure the predicate is actually defined.")
                else:
                    hint = "Ensure strict Prolog syntax. No bullets/hyphens. Predicates must be lowercase."
                _record_error(f"Prolog Error: {e}\nHint: {hint}")
            except Exception as e:
                print(f"Extension {i} | General Error: {e}")
                if "missing 3 required positional arguments" in str(e):
                    raise e
                _record_error(f"Error: {e}. Please ensure valid Prolog syntax.")
        print(f"[Prolog retry] Extension attempts also exhausted.")
    else:
        print(f"[Prolog retry] LLM says NO (or check failed) — skipping extension.")

    raise Exception("Failed to generate valid Prolog code after multiple attempts.")

