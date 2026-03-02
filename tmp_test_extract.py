import re

def extract_answer_letter(response: str) -> str:
    if not response:
        return ""

    response = response.strip()

    # Pattern 0: Bold/italic format *D.* or **D.** at the start of a line
    match = re.search(r'(?:^|\n)\s*\*{1,2}([A-D])\.?\*{1,2}', response, re.IGNORECASE)
    if match:
        print(f"Matched Pattern 0: {match.group(1)}")
        return match.group(1).upper()

    # Pattern 1: Explicit "Final Answer: X"
    match = re.search(r'(?:final\s+answer|correct\s+answer\s+is|answer\s+is)[\s\*:\-]*([A-D])\b', response, re.IGNORECASE)
    if match:
        print(f"Matched Pattern 1: {match.group(1)}")
        return match.group(1).upper()

    # Pattern 2: "Answer: X" or "Choice: X"
    match = re.search(r'(?:answer|choice)[\s\*:\-]*([A-D])\b', response, re.IGNORECASE)
    if match:
        print(f"Matched Pattern 2: {match.group(1)}")
        return match.group(1).upper()

    # Pattern 3: Standalone letter at the VERY END
    match = re.search(r'(?:\*\*|\*|\s|^)([A-D])(?:\*\*|\*|\s|\.|\)|$)', response[-10:], re.IGNORECASE)
    if match:
        print(f"Matched Pattern 3: {match.group(1)}")
        return match.group(1).upper()

    # Pattern 4: Standalone letter at the VERY START
    match = re.match(r'^[\*\s]*([A-D])(?:[\.\)]|\b)', response, re.IGNORECASE)
    if match:
        print(f"Matched Pattern 4: {match.group(1)}")
        return match.group(1).upper()

    return ""

test_text = """*D.* The difference between consumer and producer surplus is $25.
The difference between consumer and producer surplus is calculated by subtracting the producer surplus from the consumer surplus. In this case, the consumer surplus is $50 (the difference between $175 and $125), and the producer surplus is $25 (the difference between $125 and $100). Therefore, the difference between consumer and producer surplus is $50 - $25 = $25."""

print(f"Result: {extract_answer_letter(test_text)}")
