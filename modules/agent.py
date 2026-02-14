import google.generativeai as genai
import time
import os
from google.api_core import exceptions
from modules.retriever import Retriever

class Agent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        # Create the tool from the retriever method
        self.tools = [self.retriever_tool]
        self.model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), # Updated default to stable version
            tools=self.tools,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
            ),
            system_instruction="Answer concisely and use the provided context."
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    def retriever_tool(self, query: str):
        """Retrieves relevant information from the document to answer the query."""
        return self.retriever.query(query)

    def ask(self, question: str):
        retry_delay = 2
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.chat.send_message(question)
                return response.text
            except (exceptions.ServiceUnavailable, exceptions.GatewayTimeout, exceptions.DeadlineExceeded) as e:
                 if attempt < max_retries - 1:
                    print(f"Chat error ({type(e).__name__}): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                 else:
                    raise e
            except Exception as e:
                raise e
