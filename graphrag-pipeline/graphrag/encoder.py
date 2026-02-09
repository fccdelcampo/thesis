from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
import os
import glob
from .config import DOC_PATH

import asyncio

async def process_pdf_documents(kg_builder_pdf: None | SimpleKGPipeline):
    paths = glob.glob(os.path.join(DOC_PATH, "*.pdf"))
    
    for path in paths:
        print(f"Processing document: {path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=path)
        print(f"Document processed: {pdf_result}")

async def process_context(kg_builder_pdf, kg_builder_text):
    await asyncio.gather(process_pdf_documents(kg_builder_pdf))