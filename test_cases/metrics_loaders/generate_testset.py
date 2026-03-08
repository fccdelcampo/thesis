import os
from llama_parse import LlamaParse
import re
import json
import random
import pandas as pd
from google import genai
from google.genai import types

def convert_docs(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            output_md_path = os.path.join(output_dir, md_filename)
            print(f"Processing '{pdf_path}'...")
            convert_pdf_to_md(pdf_path, output_md_path)

def convert_pdf_to_md(pdf_path, output_md_path):
    # 1. Provide your LlamaCloud API Key
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    
    # 2. Initialize the parser with strict mathematical instructions
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True,
        system_prompt=(
            "You are a selective academic content extractor. Your goal is to extract ONLY the core educational material. "
            "1. SKIP: Table of Contents, Forewords, Author Biographies, Copyright pages, Indices, and Appendices. "
            "2. SKIP: 'For Dummies' branding, icons, 'Tip'/'Remember' sidebars if they are purely decorative, and footers/headers. "
            "3. EXTRACT: All main chapter content, explanations, chemical reactions, and mathematical proofs. "
            "4. FORMATTING: Transcribe the core content into clean Markdown. Use standard LaTeX ($...$) for all "
            "chemical formulas and mathematical equations. "
            "5. NO CONVERSATION: Do not explain what you are skipping. If a page contains only skipped content, "
            "return an empty string. Do not apologize."
        )
    )
    
    
    # 3. Read and process the PDF
    print(f"🔍 AI Vision is scanning '{pdf_path}'...")
    # This will take a few minutes for 250 pages, but it will print progress!
    documents = parser.load_data(pdf_path)
    
    # 4. Stitch the parsed pages together cleanly
    full_markdown = "\n\n".join([doc.text for doc in documents])
    
    # 5. Save the pristine text to a new file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        
    print(f"🎉 Success! Pristine Markdown saved to '{output_md_path}'")

def generate_resilient_testset(md_file_dir, output_jsonl, target_cases, chunks_per_prompt=6, topic: str ="general_science"):
    """Generates strict multi-hop QA pairs with an automated retry loop for rejected chunks."""
    
    # 1. Ingest the clean Markdown file
    full_text = ""
    with open(md_file_dir, "r", encoding="utf-8") as f:
        full_text += f.read()
    
    # Split into meaningful chunks (filtering out tiny layout artifacts)
    chunks_list = [c.strip() for c in full_text.split("\n\n") if len(c.strip()) > 100]
    print(f"📦 Loaded {len(chunks_list)} valid text chunks.")
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    test_cases = []
    
    # 🎯 The Prompt: Includes the 6 logic rules AND the Escape Hatch
    system_prompt = """
ROLE
You are an expert evaluation engineer specializing in First-Order Logic (FOL), deductive reasoning, and Retrieval-Augmented Generation (RAG) benchmarking.
Your task is to generate evaluation test cases that simulate Proof by Resolution using natural language rather than formal symbolic notation.
Each test case MUST require deductive logical inference across multiple independent context chunks.


OBJECTIVE
Construct a binary multi-hop reasoning test case composed of:

1. A Question
2. A Context consisting of multiple logically disconnected text chunks
3. A "yes" or "no Ground Truth Answer directly deducible from the facts and rules.

The answer MUST be derivable ONLY by resolving a chain of premises retrieved from different chunks.

A correct solution must require:
• retrieving Fact(s) and/or Rule(s) from one chunk,
• retrieving Fact(s) and/or Rule(s) from more chunks,
• applying deductive logic to derive a single unavoidable conclusion.

The question MUST be impossible to answer using any single chunk alone.


CORE LOGICAL REQUIREMENT — RESOLUTION STRUCTURE
Every generated test case MUST follow a deductive syllogistic structure equivalent to logical resolution:

Required structure:
- Chunk A: provides a factual premise about an entity or situation or concept and/or general rule, law, theorem, constraint, or causal principle..
- Chunk B: provides a factual premise about an entity or situation or concept and/or general rule, law, theorem, constraint, or causal principle.
- Optional Chunk C+: provides intermediate premises if needed.

The reasoning chain MUST correspond to one of the following deductive patterns:

• Modus Ponens  
• Modus Tollens  
• Hypothetical Syllogism  
• Proof by Resolution
• Disjunctive Syllogism  
• Transitive Classification  
• Deterministic Cause → Effect reasoning

The final answer must logically follow with certainty.


STRICT CONSTRAINTS

1. Use ONLY natural language.
   NEVER use symbolic logic notation (∀, ∃, →, ¬, variables, predicates, etc.).

2. The reasoning MUST require information retrieved from AT LEAST TWO DIFFERENT context chunks.

3. Deductive logic ONLY focusing on rule application.
   DO NOT generate abductive, probabilistic, speculative, or inductive reasoning tasks.

4. The task MUST demand multi-step logical resolution.
   Reject any case solvable through:
   - single-hop retrieval,
   - paraphrasing,
   - definition lookup,
   - commonsense reasoning.

5. The answer MUST be a single precise conclusion directly entailed by the premises.

6. ABSOLUTELY EXCLUDE:
   - numerical calculations
   - algebra, calculus, or equations
   - proof construction
   - variable manipulation
   - optimization or estimation tasks
   - person or date identification

7. Focus strictly on conceptual logic such as:
   causal consequences,
   rule application,
   or property inheritance.

8. The question MUST NOT reference the existence of chunks or source text.
   Forbidden phrases include:
   "based on the text",
   "according to the passage",
   "from the provided information".

9. All logical gaps MUST be closed using explicitly stated rules within the context.
   The answer must NOT depend on outside knowledge.

10. DO NOT request personalities, dates, explanations, summaries, lists, or reasoning steps.
    The required answer must be only the logically derived conclusion.

11. Ensure that removing ANY required chunk would make the question unanswerable.

12. The question should not be easily answerable by an LLM without symbolic logical deduction.

13. ESCAPE HATCH (MANDATORY):
If the provided chunks cannot form a strict deductive multi-hop connection,
you MUST NOT fabricate relationships.

Return:
{
  "question": null,
  "ground_truth_answer": null,
  "context": null
}


INTERNAL VALIDATION (REQUIRED BEFORE OUTPUT)
Before producing the test case, silently verify:

✓ At least two chunks are necessary  
✓ At least two deductive rules connect them  
✓ The reasoning chain is deterministic  
✓ The conclusion follows necessarily  
✓ No computation or proof construction is required  
✓ Not easily answerable by an LLM without symbolic logic


OUTPUT FORMAT
Respond ONLY with valid JSON using EXACTLY this structure:

{
  "context": "Exact concatenation of the chunks used",
  "reasoning": "The deductive reasoning chain that uses the facts and rules retrieved to conclude the answer",
  "checklist": "Check off the internal validation criteria to ensure the test case meets all requirements"
  "question": "Multi-hop deductive question or null",
  "ground_truth_answer": "Single logically entailed conclusion or null",
}"""
    
    print(f"🚀 Hunting for {target_cases} valid multi-hop connections...")
    
    attempts = 0
    valid_cases = 0
    max_attempts = target_cases * 10 # Safety limit so it doesn't spin forever
    
    # 2. The Retry Loop
    while valid_cases < target_cases and attempts < max_attempts:
        attempts += 1
        print(f"\n🔄 Attempt {attempts}: Sampling chunks and prompting the LLM...")
        try:
            # Grab a diverse batch of chunks
            index = random.randint(0, max(0, len(chunks_list) - chunks_per_prompt))
            sampled_chunks = random.sample(chunks_list[index:min(len(chunks_list), index + 30)], min(chunks_per_prompt, len(chunks_list)))
            context_block = "\n\n--- NEXT CHUNK ---\n\n".join(sampled_chunks)
            
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=f"Text Chunks:\n{context_block}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.7,
                )
            )
                       
            print(f"LLM has responded with:\n{response.text}")
            
            raw_text = response.text
            
            # Dynamically search for the JSON object boundaries
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            
            if match:
                clean_json_str = match.group(0)
                result = json.loads(clean_json_str)
            else:
                # If no JSON object brackets are found at all, force a redraw
                print(f"🔄 Attempt {attempts}: No valid JSON object found. Redrawing...")
                continue
            
            # 3. Check if the LLM used the Escape Hatch
            if not result.get("question") or str(result.get("question")).strip().lower() == "null":
                print(f"🔄 Attempt {attempts}: Chunks incompatible. Redrawing...")
                continue # Skip to the next iteration without saving
                
            # If it passed, save it!
            result["topic"] = topic
            result["context"] = context_block 
            test_cases.append(result)
            valid_cases += 1
            attempts = 0
            
            print(f"✅ Success! Found valid multi-hop connection ({valid_cases}/{target_cases})")
            
        except Exception as e:
            print(f"⚠️ Attempt {attempts} failed due to API/Parsing error: {e}")
            
    if valid_cases < target_cases:
        print(f"⚠️ Reached maximum attempts. Saving the {valid_cases} cases found.")
        
    # 4. Export to JSONL
    df = pd.DataFrame(test_cases)[['topic', 'question', 'context', 'ground_truth_answer', 'reasoning']]
    print(f"Test cases:\n{df}")
    
    # Save as JSON Lines
    df.to_json(output_jsonl, orient='records', lines=True, mode='a')
    
    print(f"🎉 Done! {len(df)} pristine multi-hop test cases saved to {output_jsonl}")
    return df

def main():
    # Step 1: Convert PDFs to clean Markdown
    input_pdf_dir = "/Users/franciscarlodelcampo/Documents/thesis/documents/science_pdfs"
    output_md_dir = "/Users/franciscarlodelcampo/Documents/thesis/documents/science_mds"
    if not os.path.exists(input_pdf_dir):
        raise ValueError("Skipping PDF conversion since input directory does not exist.")
        
    if not os.path.exists(output_md_dir):
        os.makedirs(output_md_dir)
    
    pdf_files = sorted(os.listdir(input_pdf_dir))
    md_files = sorted(os.listdir(output_md_dir))
    
    if len(md_files) <= len(pdf_files):
        if len(md_files) == 0:
            convert_docs(input_pdf_dir, output_md_dir)
    
    pdf_files = sorted(os.listdir(input_pdf_dir))
    md_files = sorted(os.listdir(output_md_dir))
    
    if len(md_files) < len(pdf_files):
        raise ValueError(f"Warning: Found {len(pdf_files)} PDFs but only {len(md_files)} Markdown files.")
    
    for pdf_file, md_file in zip(pdf_files, md_files):
        if not pdf_file.endswith(".pdf"):
            raise ValueError(f"Expected PDF files in input directory, but found '{pdf_file}'")
        if not md_file.endswith(".md"):
            raise ValueError(f"Expected Markdown files in output directory, but found '{md_file}'")
        if os.path.splitext(pdf_file)[0] + ".md" != md_file:
            raise ValueError(f"Filename mismatch: PDF '{pdf_file}' does not correspond to Markdown '{md_file}'")  
        
    
    # Step 2: Generate the resilient multi-hop test set
    output_jsonl = "/Users/franciscarlodelcampo/Documents/thesis/metrics.jsonl"
    for i, filename in enumerate(os.listdir(output_md_dir)):
        if filename.endswith(".md") and "chemistry" in filename.lower():
            md_path = os.path.join(output_md_dir, filename)
            topic = "physics" if "physics" in filename.lower() else "biology" if "biology" in filename.lower() else "chemistry" if "chemistry" in filename.lower() else "general_science"
            df = generate_resilient_testset(md_path, output_jsonl=output_jsonl, target_cases=10, topic=topic)
    print(f"Generated test cases:\n{df}")

# --- Execution ---
if __name__ == "__main__":
    main()