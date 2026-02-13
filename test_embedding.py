from modules.embedder import get_embedding
import os
from dotenv import load_dotenv

load_dotenv()

try:
    print("Testing get_embedding function...")
    embedding = get_embedding("Hello world")
    if embedding and len(embedding) > 0:
        print("SUCCESS: Embedding generated successfully.")
        print(f"Embedding length: {len(embedding)}")
    else:
        print("FAILURE: Embedding is empty.")
except Exception as e:
    print(f"FAILURE: get_embedding failed with error: {e}")
