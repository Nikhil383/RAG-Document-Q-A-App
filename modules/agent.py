import google.generativeai as genai
from modules.retriever import Retriever

class Agent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        # Create the tool from the retriever method
        self.tools = [self.retriever_tool]
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
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
        response = self.chat.send_message(question)
        return response.text
