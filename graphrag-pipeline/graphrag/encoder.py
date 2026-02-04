from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
import os
import glob
from .config import DOC_PATH, ENCODER_SYSTEM_PROMPT, ENCODER_FEW_SHOT_EXAMPLES

import ollama
import json
import asyncio

ollama_messages = [
            {'role': 'system', 'content': ENCODER_SYSTEM_PROMPT},
            
        ] + ENCODER_FEW_SHOT_EXAMPLES

def generate_with_ollama(user_prompt, most_recent_error) -> dict: 
    prompt = user_prompt if most_recent_error is None else user_prompt + "\nYour latest output had error/s. Fix them according to the following error message:\n{most_recent_error}"
    message = {'role': 'user', 'content': prompt}
    ollama_messages.append(message)
    answer = str()
    response = ollama.chat(
        model='llama3:instruct',
        messages=ollama_messages,
        options={
            'temperature': 0  
        },
        logprobs=True,
        format='json'     
    )
    
    try: 
        answer = response['message']['content']
        return answer
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
    
    
def extract_query_and_context(user_prompt: str) -> tuple[str, str]:
    most_recent_error = None
    for i in range(5):
        try:
            response = generate_with_ollama(user_prompt, most_recent_error)
            print("Ollama Response:", response)
            if response is None:
                raise ValueError("Failed to get a response from Ollama.")
            
            try:
                extracted_data = json.loads(response)
                return extracted_data["cleaned_query"], extracted_data["extracted_context"]
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                raise ValueError("Failed to decode JSON response from Ollama.")
        except Exception as e:
            most_recent_error = str(e)

async def process_pdf_documents(kg_builder_pdf: None | SimpleKGPipeline):
    paths = glob.glob(os.path.join(DOC_PATH, "*.pdf"))
    
    for path in paths:
        print(f"Processing document: {path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=path)
        print(f"Document processed: {pdf_result}")

async def process_text_context(kg_builder_text: SimpleKGPipeline, texts: list[str]):
    for text in texts:
        print(f"Processing text context: {text}")
        text_result = await kg_builder_text.run_async(text=text)
        print(f"Text context processed: {text_result}")

async def process_context(kg_builder_pdf, kg_builder_text, texts):
    await asyncio.gather(process_pdf_documents(kg_builder_pdf))
    if texts:
        await asyncio.gather(process_text_context(kg_builder_text, texts))
        
        
        
# query, context = extract_query_and_context("Francis is a student studying computer science. Like any CS major, he is interested in machine learning and artificial intelligence. What courses should he take to specialize in these fields?")
# print("Extracted Query:", query)
# print("\nExtracted Context:", context)