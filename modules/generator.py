import google.generativeai as genai

def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = f"use this context to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = genai.GenerativeModel("gemini-3-flash-preview").generate_content(prompt)
    return response.text
