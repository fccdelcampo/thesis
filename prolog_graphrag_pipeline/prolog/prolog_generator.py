import janus_swi as janus
import re
import sys

def capture_db_and_query(answer_text: str) -> tuple[str, str]:
    import re

    # Match everything after <database> until either </database>, <query>, or the end of the string
    db_match = re.search(r"<database>(.*?)(?:</database>|<query>|\Z)", answer_text, re.DOTALL)
    
    # Match everything after <query> until </query>, </reasoning_step> (common hallucination), or the end of the string
    query_match = re.search(r"<query>(.*?)(?:</query>|</reasoning_step>|\Z)", answer_text, re.DOTALL)

    if not db_match:
        raise ValueError(f"Database not found in LLM's response \n{answer_text}")
    db = db_match.group(1).strip()

    if not query_match:
        raise ValueError(f"Query not found in LLM's response: \n{answer_text}")
    query = query_match.group(1).strip()

    return {"database": db, "query": query}


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

Analyze the errors carefully. Then, rewrite the ENTIRE `<database>` and `<query>` blocks incorporating your fixes. DO NOT output partial code.
Ensure your `<database>` contains all facts and rules (using `:-`), and your `<query>` contains exactly one simple callable goal.
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

    # VALIDATION: Query must NOT contain rule definitions (:-) — those belong in <database>
    # Strip comments and quoted strings before checking, to avoid false positives
    query_stripped = re.sub(r"'[^']*'", "''", query)
    query_stripped = re.sub(r"%.*$", "", query_stripped, flags=re.MULTILINE)
    if ":-" in query_stripped:
        raise ValueError(
            "The <query> tag contains a rule definition (with ':-'), but queries must be plain goals. "
            "Rule definitions (Head :- Body.) belong inside <database> tags. "
            "The <query> tag must contain ONLY a single callable goal, e.g.: `answer(X).` or `my_predicate(foo).` "
            "Move the rule bodies to <database> and call the predicate head in <query>."
        )

    # CLEANUP: Strip inline comments (% ...) from the query
    query = re.sub(r"%.*$", "", query, flags=re.MULTILINE).strip()

    # CLEANUP: Keep only the FIRST complete Prolog statement.
    # The LLM sometimes adds multiple goals or comments after the first one.
    # A valid query is exactly one goal ending with '.'.
    first_dot = query.find('.')
    if first_dot != -1:
        query = query[:first_dot + 1].strip()

    # VALIDATION: Ensure Query ends with a period
    if not query.endswith('.'):
        query += "."

    # VALIDATION: Ensure no unquoted logic symbols exist
    combined_code = database + "\n" + query
    unquoted_code = re.sub(r"'[^']*'", "''", combined_code)
    unquoted_code = re.sub(r"%.*$", "", unquoted_code, flags=re.MULTILINE)

    # VALIDATION: MCQ Enforce answer/1
    # If the database contains 'choice(', it is a Multiple-Choice Question.
    # Therefore, the query MUST be 'answer(OptionLetter).' to unify against the choices.
    if "choice(" in unquoted_code:
        if "answer(" not in query:
            raise ValueError("Multiple-Choice constraint violation: You defined 'choice/2' facts but your <query> does not use 'answer/1'. "
                             "For MCQs, you MUST define an `answer(OptionLetter) :- ...` rule in the <database> "
                             "and your <query> MUST be exactly `answer(OptionLetter).` to unify with the correct choice.")
        
        # Check for ungrounded choice facts: e.g. choice(a, A). or choice(A, b).
        # We look for choice( followed by anything, where either argument starts with an Uppercase letter.
        if re.search(r"choice\([a-z]+,\s*[A-Z_][a-zA-Z0-9_]*\)\.", unquoted_code):
            raise ValueError("Multiple-Choice constraint violation: Your 'choice(letter, Description).' facts use UPPERCASE variables for the description (e.g., `choice(a, A).`). "
                             "The description MUST be a grounded LOWERCASE atom (e.g., `choice(a, cell_wall).`). "
                             "Uppercase variables will unify with anything and break the proof!")


    forbidden_patterns = [
        (r';', "Disjunction ';' is forbidden in s(CASP). Use multiple separate rules with the same head instead of ';'."),
        (r'=:=', "Arithmetic equality '=:=' is forbidden in s(CASP). Use '#=' for constraint arithmetic or '=' for unification."),
        (r'=\\=', "Arithmetic inequality '=\\=' is forbidden in s(CASP). Use '#=\\=' for constraint arithmetic."),
        (r'\bis\b', "Evaluation 'is' is forbidden in s(CASP). Use '#=' for constraint arithmetic assignment."),
        (r'(?<!#)<', "Standard arithmetic less-than '<' is forbidden. Use '#<' instead."),
        (r'(?<!#)>', "Standard arithmetic greater-than '>' is forbidden. Use '#>' instead."),
        (r'<=', "Standard arithmetic '<=' is forbidden. Use '#=<' instead."),
        (r'>=', "Standard arithmetic '>=' is forbidden. Use '#>=' instead."),
        (r'->', "If-then '->' is forbidden in s(CASP). Use separate rules instead."),
        (r'[~⊃∨∧]', "Unquoted logic symbol found. Wrap logical formulas in single quotes or use 'not'/'-' for negation."),
        (r'\\\\\\+', "Negation '\\+' is forbidden in s(CASP). Use 'not' for negation as failure."),
    ]

    for pattern, hint in forbidden_patterns:
        if re.search(pattern, unquoted_code):
            raise ValueError(f"{hint}")

    print(f"Iteration {i} | Extracted Database:\n{database}\n")
    print(f"Iteration {i} | Extracted Query:\n{query}\n")
    
    # --- NEW DYNAMIC WIKIDATA INJECTION ---
    # Global Config Toggle
    try:
        from .prolog_config import ENABLE_WIKIDATA_FETCHING
    except ImportError:
        try:
            from prolog_graphrag_pipeline.prolog.prolog_config import ENABLE_WIKIDATA_FETCHING
        except ImportError:
            ENABLE_WIKIDATA_FETCHING = True
            
    if getattr(sys.modules[__name__], 'ENABLE_WIKIDATA_FETCHING', ENABLE_WIKIDATA_FETCHING):
        # Parse retrieved_context to detect Wikidata Q-IDs to fetch
        qids_to_fetch = set(re.findall(r'wikidata_qid:\s*(Q\d+)', retrieved_context, re.IGNORECASE))
        
        if qids_to_fetch:
            print(f"Iteration {i} | Identified Wikidata IDs: {qids_to_fetch}")
            try:
                from prolog_graphrag_pipeline.graphrag.wikidata_retriever import WikidataRetriever
            except ImportError:
                from graphrag.wikidata_retriever import WikidataRetriever
                
            fetcher = WikidataRetriever()
            for qid in qids_to_fetch:
                print(f"Iteration {i} | Fetching active facts for {qid}...")
                # Bridge to Prolog logic engine: get assertions
                prolog_facts = fetcher.retrieve_facts(qid)
                
                # Inject facts dynamically into the active Prolog Knowledge Base string as comments
                if prolog_facts:
                    database += f"\n% --- Dynamically Retrieved Wikidata Facts for {qid} ---\n"
                    database += "\n".join([f"% {fact}" for fact in prolog_facts]) + "\n"
    # --- END INJECTION ---

    janus.query_once("unload_file(user)")
    janus.consult("user", database)
    # Run query with a 10-second hard timeout to catch infinite recursive loops
    safe_query = f"call_with_time_limit(20, ({query[:-1]}))"
    janus.query_once(safe_query)

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
                        "Example — WRONG: `answer(X) :- choice_a(X) ; choice_b(X).` "
                        "RIGHT: `answer(X) :- choice_a(X).` and `answer(X) :- choice_b(X).`")
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
            elif "time_limit_exceeded" in error_str or "Time limit" in error_str:
                hint = ("The query timed out — your rules form an INFINITE LOOP with no ground base case. "
                        "Check for circular rule chains (e.g., A :- B. and B :- A.) with nothing to terminate them. "
                        "Every recursive rule MUST bottom out at a plain ground fact (e.g., `animal(cat).` not `animal(Animal).`). "
                        "Also: bare facts like `claw.` do NOT give any predicate a value — use `is_choice_claw).` instead.")
            elif "not sufficiently instantiated" in error_str:
                hint = ("You used an UPPERCASE variable as a constant in a database fact (e.g., `choice_A, cell_wall).` where A is unbound). "
                        "Use LOWERCASE atoms for constants in facts (e.g., `choice_a, cell_wall).`). "
                        "Variables (uppercase) should only appear in rule heads/bodies where they are instantiated by matching facts.")
            elif "(:-)/2" in error_str or "Rules must be loaded from a file" in error_str:
                hint = ("You put a rule definition (Head :- Body.) inside the <query> tag. "
                        "The <query> tag must contain ONLY a plain callable goal, e.g.: `answer(X).` "
                        "Rule definitions with ':-' belong inside <database> tags. "
                        "Move the rule body to <database> and call the rule's head predicate in <query>.")
            elif "Unknown procedure" in error_str:
                hint = ("You called a predicate in a rule body that is NOT defined anywhere in the database (closed-world violation). "
                        "Every predicate that appears in a rule body MUST exist as a fact or the head of a rule in your database. "
                        "NEVER invent predicate names that have no corresponding definition. "
                        "If you use `choice(Option, Value)` in a rule body, you MUST define `choice/2` facts for every option. "
                        "Make sure your predicate arities match (e.g., if `choice/2` is defined, use `choice(Option, Value)`).")
            else:
                hint = "Ensure strict Prolog syntax. No bullets/hyphens. Predicates must be lowercase."
            _record_error(f"Prolog Error: {e}\nHint: {hint}")
        except Exception as e:
            print(f"Iteration {i} | General Error: {e}")
            if "missing 3 required positional arguments" in str(e):
                raise e  # Fatal — stop immediately
            if isinstance(e, TimeoutError):
                _record_error(f"LLM call timed out ({e}). The model may have hung. It has been reset — retry.")
            else:
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
                err_str = str(e)
                print(f"Extension {i} | Validation Error: {err_str}")
                if "Disjunction" in err_str or "';'" in err_str:
                    hint = ("Forbidden: ';' (disjunction) is not allowed in s(CASP). "
                            "Replace it with multiple rules that share the same head. "
                            "Example — WRONG: `answer(X) :- choice_a(X) ; choice_b(X).` "
                            "RIGHT: `answer(X) :- choice_a(X).` and `answer(X) :- choice_b(X).`")
                    _record_error(f"{err_str} {hint}")
                else:
                    _record_error(f"{err_str} Ensure the query is a valid Prolog predicate.")
            except janus.PrologError as e:
                print(f"Extension {i} | Prolog Error: {e}")
                error_str = str(e)
                if "Singleton variables" in error_str:
                    hint = "Fix the singleton variables by ensuring each variable appears at least twice in a rule."
                elif "Syntax error: Operator expected" in error_str:
                    hint = "Wrap mathematical/logical symbols in single quotes."
                elif "time_limit_exceeded" in error_str or "Time" in error_str:
                    hint = ("The query timed out — your rules form an infinite loop with no ground base case. "
                            "Break the cycle: ensure every recursive rule bottoms out at a grounded fact.")
                elif "not sufficiently instantiated" in error_str:
                    hint = ("You used an UPPERCASE variable as a constant in a database fact (e.g., `choice_A, cell_wall).`). "
                            "Use LOWERCASE atoms instead (e.g., `choice_a, cell_wall).`).")
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
                if isinstance(e, TimeoutError):
                    _record_error(f"LLM call timed out ({e}). The model may have hung. It has been reset — retry.")
                else:
                    _record_error(f"Error: {e}. Please ensure valid Prolog syntax.")
        print(f"[Prolog retry] Extension attempts also exhausted.")
    else:
        print(f"[Prolog retry] LLM says NO (or check failed) — skipping extension.")

    raise Exception("Failed to generate valid Prolog code after multiple attempts.")

