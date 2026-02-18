from langchain_community.chat_models import ChatOllama

def test_llm_knowledge():
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M")
    
    prompt = """
    You are a Wikidata expert. Provide the Property ID (P-code) for the following relationships.
    Output ONLY the P-code (e.g., P31). NO other text.
    
    1. Instance of:
    2. Educated at:
    3. Formula:
    4. Occupation:
    5. Opposite of:
    6. Father:
    """
    
    print("Asking LLM for Wikidata P-codes...")
    try:
        response = llm.invoke(prompt).content
        print("\nLLM Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm_knowledge()
