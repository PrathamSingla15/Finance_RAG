import os
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import warnings

import openai
import numpy as np
import pandas as pd
import faiss
from tqdm.auto import tqdm
import PyPDF2
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration constants
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
EMBEDDING_MODEL = "text-embedding-3-small"

class BasicTextSplitter:
    """Basic recursive character text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                last_period = text.rfind('.', start + self.chunk_size - 200, end)
                last_newline = text.rfind('\n', start + self.chunk_size - 200, end)
                
                # Use the latest sentence boundary found
                boundary = max(last_period, last_newline)
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
                
        return chunks

def extract_text_from_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            extracted_text = ""
            for page_num in tqdm(range(num_pages), desc=f"Parsing {Path(pdf_file_path).name}", leave=False):
                page = pdf_reader.pages[page_num]
                extracted_text += " " + page.extract_text()
            return extracted_text
    except Exception as e:
        print(f"Error reading {pdf_file_path}: {e}")
        return None

def process_pdf_directory(pdf_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """Process all PDFs in a directory and return chunks with document names."""
    pdf_directory = Path(pdf_directory)
    all_chunks = []
    
    if not pdf_directory.exists():
        raise ValueError(f"Directory {pdf_directory} does not exist")
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    text_splitter = BasicTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        extracted_text = extract_text_from_pdf(pdf_file)
        
        if extracted_text:
            chunks = text_splitter.split_text(extracted_text)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append({
                        "doc_name": pdf_file.name,
                        "content": chunk,
                        "chunk_id": f"{pdf_file.stem}_chunk_{i}"
                    })
    
    print(f"Generated {len(all_chunks)} chunks from {len(pdf_files)} documents")
    return all_chunks

class BasicVectorStore:
    """Basic vector store using OpenAI embeddings and FAISS."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, openai_client=None):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = None
        self.client = openai_client

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        all_embeddings = []
        batch_size = 100

        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.embedding_model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Handle individual texts on error
                for text in batch:
                    try:
                        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
                        all_embeddings.append(response.data[0].embedding)
                    except Exception:
                        all_embeddings.append([0.0] * 1536)  # Zero vector fallback

        return all_embeddings

    def create_index(self, chunks: List[Dict[str, str]]) -> Optional[faiss.Index]:
        """Create a FAISS vector store from document chunks."""
        if not chunks:
            print("No chunks provided to create vector store.")
            return None

        self.chunks = chunks
        chunk_contents = [chunk["content"] for chunk in chunks]

        try:
            embeddings = self.get_embeddings(chunk_contents)
            embedding_matrix = np.array(embeddings).astype("float32")

            self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            self.index.add(embedding_matrix)

            print(f"FAISS index created with {self.index.ntotal} vectors")
            return self.index

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

class BasicRetriever:
    """Basic retriever using simple vector similarity."""

    def __init__(self, vector_store: BasicVectorStore, k: int = 3):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """Retrieve top-k most similar documents."""
        if self.vector_store.index is None or self.vector_store.chunks is None:
            print("Vector store not initialized.")
            return []

        try:
            query_embedding = np.array(self.vector_store.get_embeddings([query])).astype("float32")
            distances, indices = self.vector_store.index.search(query_embedding, self.k)

            retrieved_chunks = []
            for idx in indices[0]:
                if idx != -1:  # Valid index
                    retrieved_chunks.append(self.vector_store.chunks[idx])

            return retrieved_chunks
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

class BasicRAGPipeline:
    """Basic RAG pipeline with simple retrieval and generation."""

    def __init__(self, retriever: BasicRetriever, model: str = DEFAULT_MODEL, openai_client=None):
        self.retriever = retriever
        self.model = model
        self.client = openai_client

    def generate_answer(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Generate answer using basic RAG approach."""
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query)
        
        if not retrieved_chunks:
            return "I could not find any relevant information to answer your question.", []

        # Format context
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(f"Source ({chunk['doc_name']}):\n{chunk['content']}")
        
        context_str = "\n\n---\n\n".join(context_parts)

        # Generate response
        prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Make sure your answer is relevant to the question and it is answered from the context only.
Cite the source document for each piece of information you use.

### Question: {query}

### Context: {context_str}

### Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
            )
            
            answer = response.choices[0].message.content
            return answer, retrieved_chunks
            
        except Exception as e:
            return f"Error generating response: {e}", retrieved_chunks

def load_financebench_dataset(num_samples: int = -1):
    """Load the FinanceBench dataset from Hugging Face."""
    dataset = load_dataset("yobro4619/financebench-domain-relevant")
    
    if num_samples == -1:
        return dataset['train']
    else:
        return dataset['train'].select(range(min(num_samples, len(dataset['train']))))

def process_questions_and_generate_results(rag_pipeline: BasicRAGPipeline, dataset, output_file: str):
    """Process questions from dataset and generate structured results."""
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing questions")):
        question = item['question']
        financebench_id = item.get('financebench_id', f"fb_{i}")
        answer = item.get('answer', '')
        evidence_info = item.get('evidence', [])
        
        try:
            model_response, retrieved_chunks = rag_pipeline.generate_answer(question)
            
            # Format retrieval info
            retrieval_info = []
            for chunk in retrieved_chunks:
                retrieval_info.append({
                    "doc_name": chunk["doc_name"],
                    "retrieved_chunk": chunk["content"]
                })
            
            result = {
                "financebench_id": financebench_id,
                "question": question,
                "model_response": model_response,
                "answer": answer,
                "retrieval_info": retrieval_info,
                "evidence_info": evidence_info
            }
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            result = {
                "financebench_id": financebench_id,
                "question": question,
                "model_response": f"Error: {str(e)}",
                "answer": answer,
                "retrieval_info": [],
                "evidence_info": evidence_info
            }
        
        results.append(result)
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    return results

def upload_to_huggingface(results: List[Dict], dataset_name: str, hf_token: str):
    """Upload results to Hugging Face as a new dataset."""
    try:
        df = pd.DataFrame(results)
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(dataset_name, token=hf_token, private=False)
        print(f"Successfully uploaded dataset to: {dataset_name}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

def main():
    parser = argparse.ArgumentParser(description="Basic RAG System for FinanceBench Dataset")
    parser.add_argument("--pdf_directory", type=str, required=True, help="Path to directory containing PDF files")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process (-1 for all)")
    parser.add_argument("--output_file", type=str, default="basic_rag_results.json", help="Output JSON file path")
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="Hugging Face dataset name to upload")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for text splitting")
    parser.add_argument("--retrieval_k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Setup OpenAI client
    client = openai.OpenAI(api_key=args.openai_api_key)
    
    # Load dataset
    print("Loading FinanceBench dataset...")
    dataset = load_financebench_dataset(args.num_samples)
    print(f"Loaded {len(dataset)} questions from dataset")
    
    # Process PDFs
    print("Processing PDF documents...")
    chunks = process_pdf_directory(
        args.pdf_directory, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap
    )
    
    # Initialize basic RAG components
    print("Initializing basic RAG system...")
    vector_store = BasicVectorStore(embedding_model=args.embedding_model, openai_client=client)
    index = vector_store.create_index(chunks)
    
    if index is None:
        print("Failed to create vector index. Exiting.")
        return
    
    retriever = BasicRetriever(vector_store, k=args.retrieval_k)
    rag_pipeline = BasicRAGPipeline(retriever, model=args.model, openai_client=client)
    
    # Process questions and generate results
    print("Processing questions and generating answers...")
    results = process_questions_and_generate_results(rag_pipeline, dataset, args.output_file)
    
    # Upload to Hugging Face
    print("Uploading results to Hugging Face...")
    upload_to_huggingface(results, args.hf_dataset_name, args.hf_token)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()