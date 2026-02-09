from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
import os
import glob
from .config import DOC_PATH, ENCODER_SYSTEM_PROMPT, ENCODER_FEW_SHOT_EXAMPLES

import ollama
import json
import asyncio

# Stateless message template
BASE_OLLAMA_MESSAGES = [
            {'role': 'system', 'content': ENCODER_SYSTEM_PROMPT},
            
        ] + ENCODER_FEW_SHOT_EXAMPLES

def generate_with_ollama(messages) -> str | None: 
    try:
        response = ollama.chat(
            model='llama3:instruct',
            messages=messages,
            options={
                'temperature': 0  
            },
            format='json'     
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
    
    
def extract_query_and_context(user_prompt: str) -> tuple[str, str]:
    most_recent_error = None
    
    # Create a fresh conversation history for this specific extraction attempt
    current_messages = list(BASE_OLLAMA_MESSAGES)

    for i in range(5):
        try:
            # Construct the prompt
            prompt_content = user_prompt
            if most_recent_error:
                prompt_content += f"\nYour latest output had error/s. Fix them according to the following error message:\n{most_recent_error}"
            

            turn_message = {'role': 'user', 'content': prompt_content}
            
            current_messages.append(turn_message)
            
            response = generate_with_ollama(current_messages)
            print("Ollama Response:", response) # verbose
            
            if response is None:
                raise ValueError("Failed to get a response from Ollama.")
            
            try:
                extracted_data = json.loads(response)
                return extracted_data["question"], extracted_data["context"]
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                current_messages.append({'role': 'assistant', 'content': response})
                most_recent_error = f"JSON Decode Error: {e}"
                
        except Exception as e:
            most_recent_error = str(e)
            
    raise ValueError(f"Failed to extract query and context after 5 retries. Last error: {most_recent_error}")

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