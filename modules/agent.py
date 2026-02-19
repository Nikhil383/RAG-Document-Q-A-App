import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from modules.retriever import Retriever


load_dotenv()


class Agent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

        # Ensure LangChain's Google client can see the API key even if the env
        # variable name follows this project's existing convention.
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

        # Wrap the existing FAISS-based retriever as a LangChain tool.
        self.tools = [self._build_retriever_tool()]

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=500,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an agentic RAG assistant for answering questions "
                        "about an uploaded PDF. When helpful, call the retrieval tool "
                        "to get relevant context and ground your answers in that text. "
                        "Cite or summarize the retrieved context rather than guessing."
                    ),
                ),
                ("human", "{input}"),
            ]
        )

        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)

    def _build_retriever_tool(self):
        retriever = self.retriever

        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve the most relevant passages from the uploaded PDF for a natural-language question."""

            return retriever.query(query)

        return retrieve_context

    def ask(self, question: str) -> str:
        """Ask the LangChain agent a question and return its response text."""

        max_retries = 3
        last_error: Exception | None = None

        for _ in range(max_retries):
            try:
                result: Dict[str, Any] = self.agent_executor.invoke({"input": question})
                return str(result.get("output", "")).strip()
            except Exception as e:
                last_error = e

        # If we get here, all retries failed.
        if last_error is not None:
            raise last_error
        raise RuntimeError("Agent execution failed without a specific error.")
