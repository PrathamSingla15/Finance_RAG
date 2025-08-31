import os
import json
import warnings
import faiss
from datetime import datetime

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_PROJECT'] = "Build RAG locally"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_TRACING_V2'] = "true"

def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def create_documents_from_corpus(corpus_data):
    """Convert corpus data to LangChain Document objects"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    for item in corpus_data:
        full_text = f"Title: {item['title']}\n\nContent: {item['text']}"

        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'source_id': item['_id'],
                    'title': item['title'],
                    'chunk_id': f"{item['_id']}_chunk_{i}"
                }
            )
            documents.append(doc)

    return documents

def setup_vector_store(documents):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

    sample_embedding = embeddings.embed_query("sample text")
    embedding_dim = len(sample_embedding)

    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=documents)

    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(retriever):
    prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.

    ### Question: {question}

    ### Context: {context}

    ### Answer:
    """

    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    return chain

def save_results_to_jsonl(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')

def main():
    print("Loading corpus and queries...")

    corpus_data = load_jsonl('./finder_corpus.jsonl')
    queries_data = load_jsonl('./finder_queries.jsonl')

    MAX_CORPUS_SAMPLES = 100  
    MAX_QUERY_SAMPLES = 10    

    corpus_data = corpus_data[:MAX_CORPUS_SAMPLES]
    queries_data = queries_data[:MAX_QUERY_SAMPLES]

    print(f"Loaded {len(corpus_data)} corpus documents and {len(queries_data)} queries")

    print("Creating documents and setting up vector store...")
    documents = create_documents_from_corpus(corpus_data)

    vector_store = setup_vector_store(documents)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3}
    )

    rag_chain = create_rag_chain(retriever)

    print(f"Vector store created with {vector_store.index.ntotal} document chunks")

    results = []
    print("\nProcessing queries...")

    for i, query_item in enumerate(queries_data):
        query_id = query_item['_id']
        query_text = query_item['text']
        query_title = query_item.get('title', '')

        print(f"\nProcessing query {i+1}/{len(queries_data)}: {query_id}")
        print(f"Query: {query_text}")

        try:
            answer = rag_chain.invoke(query_text)

            retrieved_docs = retriever.invoke(query_text)
            retrieved_sources = [doc.metadata.get('source_id', 'unknown') for doc in retrieved_docs]

            result = {
                'query_id': query_id,
                'query_title': query_title,
                'query_text': query_text,
                'answer': answer,
                'retrieved_sources': retrieved_sources,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            print(f"Answer: {answer}")
            print(f"Retrieved sources: {retrieved_sources}")

        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            result = {
                'query_id': query_id,
                'query_title': query_title,
                'query_text': query_text,
                'answer': f"Error: {str(e)}",
                'retrieved_sources': [],
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

    output_file = f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    save_results_to_jsonl(results, output_file)

    print(f"\n\nResults saved to {output_file}")
    print(f"Processed {len(results)} queries successfully")

if __name__ == "__main__":
    main()

