"""Test script for RAG application"""
import os
import sys

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    try:
        from modules.retriever.vector_store import VectorStore
        print("  [OK] VectorStore")
    except Exception as e:
        print(f"  [FAIL] VectorStore: {e}")
        return False
    
    try:
        from modules.retriever.graph_retriever import GraphRetriever
        print("  [OK] GraphRetriever")
    except Exception as e:
        print(f"  [FAIL] GraphRetriever: {e}")
        return False
    
    try:
        from modules.retriever.hybrid_retriever import HybridRetriever
        print("  [OK] HybridRetriever")
    except Exception as e:
        print(f"  [FAIL] HybridRetriever: {e}")
        return False
    
    try:
        from modules.graph.neo4j_store import Neo4jStore
        print("  [OK] Neo4jStore")
    except Exception as e:
        print(f"  [FAIL] Neo4jStore: {e}")
        return False
    
    try:
        from modules.ingestion.chunker import chunk_docs
        print("  [OK] chunk_docs")
    except Exception as e:
        print(f"  [FAIL] chunk_docs: {e}")
        return False
    
    try:
        from modules.ingestion.loader import load_document
        print("  [OK] load_document")
    except Exception as e:
        print(f"  [FAIL] load_document: {e}")
        return False
    
    try:
        from modules.ingestion.graph_builder import GraphBuilder
        print("  [OK] GraphBuilder")
    except Exception as e:
        print(f"  [FAIL] GraphBuilder: {e}")
        return False
    
    try:
        from modules.llm.llm import get_llm
        print("  [OK] get_llm")
    except Exception as e:
        print(f"  [FAIL] get_llm: {e}")
        return False
    
    try:
        from langraph_agent import build_graph
        print("  [OK] build_graph")
    except Exception as e:
        print(f"  [FAIL] build_graph: {e}")
        return False
    
    try:
        from app import app
        print("  [OK] Flask app")
    except Exception as e:
        print(f"  [FAIL] Flask app: {e}")
        return False
    
    return True


def test_env():
    """Test environment configuration"""
    print("\nTesting environment...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    checks = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    }
    
    all_ok = True
    for key, value in checks.items():
        if value:
            masked = value[:4] + "..." if len(value) > 4 else "***"
            print(f"  [OK] {key}: {masked}")
        else:
            print(f"  [FAIL] {key}: NOT SET")
            all_ok = False
    
    return all_ok


def test_chunker():
    """Test document chunking"""
    print("\nTesting chunker...")
    
    try:
        from langchain_core.documents import Document
        from modules.ingestion.chunker import chunk_docs
        
        # Create test document
        doc = Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test.txt", "page": 1}
        )
        
        chunks = chunk_docs([doc], chunk_size=200, chunk_overlap=20)
        
        if len(chunks) > 0:
            print(f"  [OK] Created {len(chunks)} chunks")
            print(f"    First chunk: {len(chunks[0].page_content)} chars")
            return True
        else:
            print("  [FAIL] No chunks created")
            return False
    
    except Exception as e:
        print(f"  [FAIL] Chunker test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("RAG Application Tests")
    print("=" * 50)
    
    results = {
        "Imports": test_imports(),
        "Environment": test_env(),
        "Chunker": test_chunker(),
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    print("=" * 50)
    
    if all_passed:
        print("\nAll tests passed! Ready to run the app.")
        print("\nTo start the server:")
        print("  uv run python app.py")
        print("\nThen open: http://localhost:5000")
        return 0
    else:
        print("\nSome tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
