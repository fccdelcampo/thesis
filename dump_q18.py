import json

lines = open('logs/prolog_graphrag_logs/openbookqa_qa.jsonl', 'r', encoding='utf-8').readlines()[-16:]
out = []
for i in [1, 8]:
    obj = json.loads(lines[i])
    out.append(f"--- Q{i} ---")
    out.append(f"Question: {obj.get('question', '')[:100]}")
    out.append(f"Answer: {obj.get('answer', '')}")
    out.append(f"Context snippet: {str(obj.get('contexts', ''))[:1000]}")
    out.append("\n-----------------------\n")

with open('dump_q18.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
