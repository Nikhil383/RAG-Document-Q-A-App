from google import genai
from google.genai import types

client = genai.Client()

SYSTEM = (
    "Answer using ONLY the context provided. "
    "If answer is not in the context, say 'I don't know'. "
    "Cite sources like [1], [2]."
)

def build_prompt(q, chunks):
    ctx = ""
    for i, c in enumerate(chunks):
        ctx += f"[{i+1}] {c['text']}\n\n"
    return f"{SYSTEM}\n\nCONTEXT:\n{ctx}\nQuestion: {q}\nAnswer:"

def generate_answer(q, chunks):
    p = build_prompt(q, chunks)
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=p,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return r.text
