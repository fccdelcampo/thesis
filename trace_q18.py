from prolog_graphrag_pipeline.main_driver import run_pipeline

q1 = "Which is recyclable A. fiberglass B. soda can C. Styrofoam food container D. motor oil"
q8 = "A refraction of light that splits into all different colors often occurs after A. clouds B. rain C. water droplets form D. the sun goes down"

print("Running Q1...")
res1 = run_pipeline(q1, "x")
print("Q1 Explanation:\n", res1.get("prolog_explanation", "None"))

print("\nRunning Q8...")
res8 = run_pipeline(q8, "x")
print("Q8 Explanation:\n", res8.get("prolog_explanation", "None"))

with open('debug_q1_q8.txt', 'w', encoding='utf-8') as f:
    f.write("Q1 Final Answer:\n" + res1.get("answer", "") + "\n\n")
    f.write("Q1 Prolog:\n" + res1.get("prolog_explanation", "") + "\n\n")
    f.write("Q8 Final Answer:\n" + res8.get("answer", "") + "\n\n")
    f.write("Q8 Prolog:\n" + res8.get("prolog_explanation", "") + "\n\n")
