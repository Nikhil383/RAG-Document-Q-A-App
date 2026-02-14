import google.generativeai as genai
import os
import time
from google.api_core import exceptions
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    retry_delay = 5  # Start with 5 seconds
    max_retries = 6  # Will wait up to approx 5+10+20+40+80+160 = 315 seconds
    
    for attempt in range(max_retries):
        try:
            embed = genai.embed_content(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"), content=text)
            time.sleep(1) # Rate limit padding
            return embed["embedding"]
        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.GatewayTimeout, exceptions.DeadlineExceeded) as e:
            if attempt < max_retries - 1:
                print(f"Temporary error ({type(e).__name__}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Max retries exceeded for error: {e}")
                raise
        except Exception as e:
            # For other exceptions, we might want to fail immediately or handle differently
            # It's generally good to log the specific error
            print(f"Embedding failed with error: {e}")
            raise e
