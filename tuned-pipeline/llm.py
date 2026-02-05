from config import LLM_MESSAGES
import ollama

def generate(user_question: str):
    try:
        answer = dict()
        response = ollama.chat(
                model='llama3.1:8b-instruct-q4_K_M',
                messages=LLM_MESSAGES + [{"role": "user", "content": user_question}],
                options={
                    'temperature': 0.4,
                    'keep_alive': 0,
                    'top_k': 50,
                    'top_p': 0.9
                },
                logprobs=True     
            )
        answer['text_answer'] = response['message']['content']
        answer['logprobs'] = response.get('logprobs', None)
        return answer
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
        