import ollama

print("Asking Ollama (with confidence scores)...")
response = ollama.chat(
    model='llama3:instruct',
    messages=[
        {'role': 'user', 'content': 'Who are you?'}
    ],
    options={
        'temperature': 0  
    },
    logprobs=True     
)

print(f"Answer: {response['message']['content']}\n")

if 'logprobs' in response:
    print("--- Confidence Scores (First 5 Tokens) ---")
    for token_data in response['logprobs'][:5]:
        token_str = token_data.get('token', '')
        score = token_data.get('logprob', -999) 
        
        # A score close to 0 (e.g., -0.01) is VERY confident.
        # A score like -2.5 is LOW confidence.
        print(f"Token: '{token_str}' | Logprob: {score:.4f}")
else:
    print("Logprobs not found. Ensure your Ollama server version is up to date.")