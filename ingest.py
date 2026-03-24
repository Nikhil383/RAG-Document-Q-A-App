import os
import sys
import argparse
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf is not installed. PDF parsing will fail. Please run `uv pip install pypdf` if you want to upload PDFs.")


def process_document(file_path):
    """Router to handle both text files and PDFs."""
    if file_path.lower().endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except NameError:
            raise ImportError("pypdf is required for PDF documents. Please run `uv pip install pypdf`.")
    else:
        # Assumes a plain text format like .txt, .md, .csv
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def ingest_data(file_path):
    """
    Extracts text from a user document, chunks it, and transforms the chunks 
    into a persistent Knowledge Graph stored in Neo4j.
    """
    load_dotenv()
    
    # Check credentials
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set.")
    
    try:
        # Connect to DB
        graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
    except Exception as e:
        print(f"Skipping ingestion: Local Neo4j connection failed ({e})")
        return
        
    print("Neo4j connection established.")
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return
        
    print(f"Loading document: {file_path} ...")
    text = process_document(file_path)
        
    if not text.strip():
        print("[Error] No text could be extracted from the file.")
        return

    # Split the document into manageable chunks to prevent exceeding LLM context limits
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    documents = splitter.create_documents([text])
    
    print(f"Document split into {len(documents)} chunks for processing.")
    
    # Initialize the Gemini model suitable for parsing
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Use LLMGraphTransformer to pull entities and relationships from the text
    transformer = LLMGraphTransformer(llm=llm)
    
    print("Extracting nodes and relationships via Gemini... This may take a minute based on file size.")
    graph_documents = transformer.convert_to_graph_documents(documents)
    
    node_count = sum(len(doc.nodes) for doc in graph_documents)
    edge_count = sum(len(doc.relationships) for doc in graph_documents)
    print(f"\nExtraction successful! Created {node_count} nodes and {edge_count} relationships.")
    
    # Merge the graph documents straight into Neo4j
    print("Writing graph out to Neo4j...")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    print("Ingestion complete. You can now use graph_agent.py to query this knowledge.")

if __name__ == "__main__":
    # Built-in Argument Parser to ensure robust CLI support
    parser = argparse.ArgumentParser(description="Ingest user documents into Neo4j Knowledge Graph via Gemini.")
    parser.add_argument("--file", "-f", type=str, help="Path to the document (.txt, .md, .pdf) to ingest.")
    args = parser.parse_args()
    
    target_file = args.file
    
    # Fallback to interactive prompt if CLI argument wasn't provided
    if not target_file:
        print("--- Node & Relationship Extractor ---")
        target_file = input("Enter the absolute or relative path to the document you want to ingest: ").strip()
        
        # Remove quotes around path if dragged-and-dropped in the terminal
        target_file = target_file.strip("'\"")
        
    ingest_data(target_file)
