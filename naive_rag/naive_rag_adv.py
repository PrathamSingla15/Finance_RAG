import os
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import glob
import pickle

import openai
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagReranker
from tqdm.auto import tqdm
import PyPDF2
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

# Configuration constants
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-base'

def extract_text_from_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            extracted_text = ""
            for page_num in tqdm(range(num_pages), desc=f"Parsing {Path(pdf_file_path).name}"):
                page = pdf_reader.pages[page_num]
                extracted_text += " " + page.extract_text()
            return extracted_text
    except FileNotFoundError:
        print(f"Error: File '{pdf_file_path}' not found.")
        return None
    except PyPDF2.errors.PdfReadError:
        print(f"Error: Could not read PDF file '{pdf_file_path}'. It might be corrupted or encrypted.")
        return None

def semantic_chunking(text: str, max_chunk_size: int = 2000, min_chunk_size: int = 500) -> List[str]:
    """Split text into semantic chunks."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk.strip() and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            elif current_chunk.strip():
                sentences = paragraph.split('. ')
                combined = current_chunk + '\n\n' + sentences[0]
                if len(combined) <= max_chunk_size:
                    current_chunk = combined
                    if len(sentences) > 1:
                        paragraph = '. '.join(sentences[1:])
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            else:
                current_chunk = paragraph

            if len(current_chunk) > max_chunk_size:
                sentences = current_chunk.split('. ')
                temp_chunk = ""

                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 2 <= max_chunk_size:
                        temp_chunk += sentence + ". " if sentence != sentences[-1] else sentence
                    else:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + ". " if sentence != sentences[-1] else sentence

                current_chunk = temp_chunk
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    if len(chunks) > 1:
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]

    return chunks

def process_pdf_directory(pdf_directory: str) -> List[Dict[str, str]]:
    """Process all PDFs in a directory and return chunks with document names."""
    pdf_directory = Path(pdf_directory)
    all_chunks = []
    
    if not pdf_directory.exists():
        raise ValueError(f"Directory {pdf_directory} does not exist")
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        print(f"Processing: {pdf_file.name}")
        extracted_text = extract_text_from_pdf(pdf_file)
        
        if extracted_text:
            chunks = semantic_chunking(extracted_text)
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append({
                        "doc_name": pdf_file.name,
                        "content": chunk
                    })
    
    print(f"Generated {len(all_chunks)} chunks from {len(pdf_files)} documents")
    return all_chunks

class VectorStore:
    """Handles vector embeddings and FAISS index operations."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, embeddings_dir: str = "embeddings"):
        self.embedding_model_name = embedding_model
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Load embedding model and tokenizer
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model, torch_dtype=torch.float32)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.index = None
        self.chunks = None
        
        print(f"Embedding model loaded on device: {self.device}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts with batch processing."""
        all_embeddings = []
        batch_size = 32  # Reduced batch size for BGE-M3
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
                batch = texts[i:i + batch_size]
                
                try:
                    # Tokenize batch
                    encoded_input = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt',
                        max_length=8192  # BGE-M3 max length
                    )
                    
                    # Move to device
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                    
                    # Get embeddings
                    model_output = self.model(**encoded_input)
                    
                    # Pool embeddings (mean pooling)
                    token_embeddings = model_output[0]
                    attention_mask = encoded_input['attention_mask']
                    
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Fallback: process individually
                    for text in batch:
                        try:
                            encoded_input = self.tokenizer(
                                [text], 
                                padding=True, 
                                truncation=True, 
                                return_tensors='pt',
                                max_length=8192
                            )
                            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                            
                            model_output = self.model(**encoded_input)
                            token_embeddings = model_output[0]
                            attention_mask = encoded_input['attention_mask']
                            
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                            embedding = sum_embeddings / sum_mask
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                            
                            all_embeddings.append(embedding.cpu().numpy().tolist()[0])
                        except Exception as individual_error:
                            print(f"Skipping problematic text: {individual_error}")
                            # Create zero embedding with correct dimensions
                            all_embeddings.append([0.0] * 1024)  # BGE-M3 embedding dimension

        return all_embeddings

    def save_embeddings(self, chunks: List[Dict[str, str]], embeddings: List[List[float]]):
        """Save chunks and embeddings to disk."""
        embeddings_file = self.embeddings_dir / "embeddings.pkl"
        chunks_file = self.embeddings_dir / "chunks.pkl"
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        with open(chunks_file, 'wb') as f:
            pickle.dump(chunks, f)
        
        print(f"Embeddings and chunks saved to {self.embeddings_dir}")

    def load_embeddings(self) -> Tuple[Optional[List[Dict[str, str]]], Optional[List[List[float]]]]:
        """Load chunks and embeddings from disk."""
        embeddings_file = self.embeddings_dir / "embeddings.pkl"
        chunks_file = self.embeddings_dir / "chunks.pkl"
        
        if embeddings_file.exists() and chunks_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                
                with open(chunks_file, 'rb') as f:
                    chunks = pickle.load(f)
                
                print(f"Loaded {len(chunks)} chunks and embeddings from {self.embeddings_dir}")
                return chunks, embeddings
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return None, None
        else:
            print("No existing embeddings found.")
            return None, None

    def create_index(self, chunks: List[Dict[str, str]]) -> Optional[faiss.Index]:
        """Create a FAISS vector store from document chunks."""
        if not chunks:
            print("No chunks were provided to create a vector store.")
            return None

        # Try to load existing embeddings
        loaded_chunks, loaded_embeddings = self.load_embeddings()
        
        if loaded_chunks is not None and loaded_embeddings is not None:
            # Check if chunks match (simple comparison by length and first chunk content)
            if (len(loaded_chunks) == len(chunks) and 
                len(loaded_chunks) > 0 and len(chunks) > 0 and
                loaded_chunks[0]['content'] == chunks[0]['content']):
                
                print("Using existing embeddings.")
                self.chunks = loaded_chunks
                embeddings = loaded_embeddings
            else:
                print("Chunks have changed. Regenerating embeddings.")
                self.chunks = chunks
                chunk_contents = [chunk["content"] for chunk in chunks]
                embeddings = self.get_embeddings(chunk_contents)
                self.save_embeddings(chunks, embeddings)
        else:
            print("Generating new embeddings.")
            self.chunks = chunks
            chunk_contents = [chunk["content"] for chunk in chunks]
            embeddings = self.get_embeddings(chunk_contents)
            self.save_embeddings(chunks, embeddings)

        try:
            embedding_matrix = np.array(embeddings).astype("float32")
            
            # Use inner product (cosine similarity) for normalized embeddings
            self.index = faiss.IndexFlatIP(embedding_matrix.shape[1])
            self.index.add(embedding_matrix)

            print("FAISS index created successfully.")
            print(f"Index size: {self.index.ntotal} vectors")
            return self.index

        except Exception as e:
            print(f"An error occurred while creating the vector store: {e}")
            return None

class RetrievalSystem:
    """Handles document retrieval and re-ranking."""

    def __init__(self, vector_store: VectorStore, cross_encoder_model: str = CROSS_ENCODER_MODEL):
        self.vector_store = vector_store
        print(f"Loading reranker model: {cross_encoder_model}")
        self.reranker = FlagReranker(cross_encoder_model, use_fp16=False)
        print("Reranker model loaded successfully.")

    def retrieve_and_rerank(self, query: str, k_initial: int = 10, k_final: int = 3) -> List[Dict[str, str]]:
        """Retrieve documents and re-rank them using cross-encoder."""
        if self.vector_store.index is None or self.vector_store.chunks is None:
            print("Vector store not initialized properly.")
            return []

        query_embedding = np.array(self.vector_store.get_embeddings([query])).astype("float32")
        distances, indices = self.vector_store.index.search(query_embedding, k_initial)

        initial_candidates = [self.vector_store.chunks[i] for i in indices[0]]
        
        # Prepare pairs for reranking
        pairs = [[query, candidate["content"]] for candidate in initial_candidates]
        scores = self.reranker.compute_score(pairs)
        
        # Handle both single score and list of scores
        if not isinstance(scores, list):
            scores = [scores]

        scored_candidates = list(zip(scores, initial_candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        reranked_chunks = [candidate for score, candidate in scored_candidates[:k_final]]

        return reranked_chunks

class QueryExpander:
    """Handles query expansion for better retrieval."""

    def __init__(self, model: str = DEFAULT_MODEL, openai_client=None):
        self.model = model
        self.client = openai_client

    def expand_query(self, query: str) -> List[str]:
        """Generate alternative phrasings of the original question."""
        
        system_prompt = """You are a helpful assistant who is an expert in finance and accounting.
        Your task is to generate 3 alternative questions that are semantically similar to the original question.
        The questions should be phrased differently to maximize the chance of matching relevant documents."""

        prompt = f"""
        Original question: {query}

        Provide the 3 alternative questions, each on a new line.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.95,
        )

        expanded_queries = response.choices[0].message.content.strip().split("\n")
        all_queries = [query] + [q.strip() for q in expanded_queries if q.strip()]

        return list(set(all_queries))

class RAGPipeline:
    """Main RAG pipeline that combines all components."""

    def __init__(self, vector_store: VectorStore, retrieval_system: RetrievalSystem,
                 query_expander: QueryExpander, model: str = DEFAULT_MODEL, openai_client=None):
        self.vector_store = vector_store
        self.retrieval_system = retrieval_system
        self.query_expander = query_expander
        self.model = model
        self.client = openai_client

    def generate_answer(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """The full, improved RAG pipeline."""
        expanded_queries = self.query_expander.expand_query(query)
        all_retrieved_chunks = []
        
        for q in expanded_queries:
            retrieved = self.retrieval_system.retrieve_and_rerank(q, k_initial=10, k_final=3)
            all_retrieved_chunks.extend(retrieved)

        unique_chunks = {chunk['content']: chunk for chunk in all_retrieved_chunks}.values()
        
        if not unique_chunks:
            return "I could not find any relevant information to answer your question.", []
        
        # Final reranking with original query
        final_rerank_pairs = [[query, chunk["content"]] for chunk in unique_chunks]
        final_scores = self.retrieval_system.reranker.compute_score(final_rerank_pairs)
        
        # Handle both single score and list of scores
        if not isinstance(final_scores, list):
            final_scores = [final_scores]

        scored_final_chunks = list(zip(final_scores, unique_chunks))
        scored_final_chunks.sort(key=lambda x: x[0], reverse=True)

        top_chunks = [chunk for score, chunk in scored_final_chunks[:3]]

        context_str = "\n\n---\n\n".join([f"Source ({chunk['doc_name']}):\n{chunk['content']}" for chunk in top_chunks])

        system_prompt = """You are a financial analyst assistant. Answer the user's question based *only* on the provided context.
        Be concise and to the point. If the answer is not available in the context, say so.
        Cite the source document for each piece of information you use."""

        prompt = f"""
        Context:
        {context_str}

        Question: {query}

        Answer:
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.95,
        )

        return response.choices[0].message.content, top_chunks

def load_financebench_dataset(num_samples: int = -1):
    """Load the FinanceBench dataset from Hugging Face."""
    dataset = load_dataset("yobro4619/financebench-domain-relevant")
    
    if num_samples == -1:
        return dataset['train']
    else:
        return dataset['train'].select(range(min(num_samples, len(dataset['train']))))

def process_questions_and_generate_results(rag_pipeline: RAGPipeline, dataset, output_file: str):
    """Process questions from dataset and generate structured results."""
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing questions")):
        question = item['question']
        financebench_id = item.get('financebench_id', f"fb_{i}")
        answer = item.get('answer', '')
        evidence_info = item.get('evidence', [])
        
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
        
        results.append(result)
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    return results

def upload_to_huggingface(results: List[Dict], dataset_name: str, hf_token: str):
    """Upload results to Hugging Face as a new dataset."""
    try:
        # Convert to Hugging Face dataset format
        df = pd.DataFrame(results)
        dataset = Dataset.from_pandas(df)
        
        # Push to Hugging Face Hub
        dataset.push_to_hub(dataset_name, token=hf_token, private=False)
        print(f"Successfully uploaded dataset to: {dataset_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG System for FinanceBench Dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to directory containing PDF files")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process (-1 for all)")
    parser.add_argument("--output_file", type=str, default="rag_results.json", help="Output JSON file path")
    parser.add_argument("--hf_dataset", type=str, required=True, help="Hugging Face dataset name to upload")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL, help="Embedding model to use")
    parser.add_argument("--cross_encoder_model", type=str, default=CROSS_ENCODER_MODEL, help="Cross-encoder model for re-ranking")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings", help="Directory to store/load embeddings")
    
    args = parser.parse_args()
    
    # Setup OpenAI client
    client = openai.OpenAI(api_key=args.openai_key)
    
    # Load dataset
    print("Loading FinanceBench dataset...")
    dataset = load_financebench_dataset(args.num_samples)
    print(f"Loaded {len(dataset)} questions from dataset")
    
    # Process PDFs
    print("Processing PDF documents...")
    chunks = process_pdf_directory(args.data)
    
    # Initialize RAG components
    print("Initializing RAG system...")
    vector_store = VectorStore(embedding_model=args.embedding_model, embeddings_dir=args.embeddings_dir)
    index = vector_store.create_index(chunks)
    
    if index is None:
        print("Failed to create vector index. Exiting.")
        return
    
    retrieval_system = RetrievalSystem(vector_store, cross_encoder_model=args.cross_encoder_model)
    query_expander = QueryExpander(model=args.model, openai_client=client)
    rag_pipeline = RAGPipeline(vector_store, retrieval_system, query_expander, 
                              model=args.model, openai_client=client)
    
    # Process questions and generate results
    print("Processing questions and generating answers...")
    results = process_questions_and_generate_results(rag_pipeline, dataset, args.output_file)
    
    # Upload to Hugging Face
    print("Uploading results to Hugging Face...")
    upload_to_huggingface(results, args.hf_dataset_name, args.hf_token)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
