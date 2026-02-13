import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models_to_test = ["models/gemini-embedding-001", "models/text-embedding-004"]

for model in models_to_test:
    print(f"Testing model: {model}")
    try:
        genai.embed_content(model=model, content="Hello world")
        print(f"SUCCESS: {model} works.")
    except Exception as e:
        print(f"FAILURE: {model} failed with error: {e}")
