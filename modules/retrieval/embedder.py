import logging
import os
import time
from typing import List, Union

import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure genai if not already configured
if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single string.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding
    """
    embeddings = get_embeddings([text])
    return embeddings[0]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of strings in batches.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of embeddings
    """
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    retry_delay = 5
    max_retries = 5
    
    # Process in batches if necessary (Gemini API has limits)
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        success = False
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=batch,
                    task_type="retrieval_document" if len(batch) > 1 else "retrieval_query"
                )
                
                if "embedding" in result:
                    # Single result
                    all_embeddings.append(result["embedding"])
                else:
                    # Multiple results
                    all_embeddings.extend(result["embeddings"])
                
                success = True
                # Small delay to respect rate limits if not hitting them
                time.sleep(0.5)
                break
                
            except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit or service error: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Max retries exceeded for embedding: {e}")
                    raise
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise
        
        if not success:
            raise RuntimeError("Failed to generate embeddings")
            
    return all_embeddings
