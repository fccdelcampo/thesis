import json

incorrect = []
with open('logs/prolog_graphrag_logs/mmlu_qa.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('answer') != data.get('correct_answer'):
            incorrect.append(data)

with open('mmlu_failure_analysis.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Incorrect: {len(incorrect)}\n")
    for d in incorrect[-20:]:  # Look at the last twenty
        query_text = (d.get('query') or d.get('question') or '').replace('\n', ' ')
        out.write(f"ID: {d.get('test_case_id')} | Q: {query_text[:100]}...\n")
        prolog = d.get('prolog_code')
        prolog_snippet = (prolog or 'None').replace('\n', ' ')[:200]
        out.write(f"  Ans: {d.get('answer')} | Correct: {d.get('correct_answer')} | Prolog OK: {d.get('prolog_verified')}\n")
        out.write(f"  Prolog Code: {prolog_snippet}...\n")
        out.write("-" * 50 + "\n")
