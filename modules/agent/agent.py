import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from modules.retrieval.retriever import Retriever


load_dotenv()


class Agent:
    def __init__(
        self,
        retriever: Retriever,
        session_id: Optional[str] = None
    ):
        """Initialize the RAG agent.

        Args:
            retriever: The document retriever instance
            session_id: Optional session identifier for context
        """
        self.retriever = retriever
        self.session_id = session_id

        # Ensure LangChain's Google client can see the API key
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

        # Build tools
        self.tools = self._build_tools()

        # Modern default model
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=1000,
        )

        # Prompt with memory placeholder
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful and intelligent Agentic RAG assistant. "
                        "Your goal is to answer questions accurately using the provided documents. "
                        "When you need specific information, use the 'retrieve_context' tool. "
                        "You can also use 'list_documents' to see what's available. "
                        "\n\nGuidelines:\n"
                        "1. Always cite source documents and page numbers (e.g., [Page 5 from report.pdf]).\n"
                        "2. If the answer isn't in the context, say so gracefully. Do not hallucinate.\n"
                        "3. Maintain a professional yet friendly tone.\n"
                        "4. If historical context is needed, refer to the chat history provided."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,
            handle_parsing_errors=True
        )

    def _build_tools(self):
        retriever = self.retriever

        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve relevant passages from the uploaded PDFs for the given question.
            Use this when you need facts from the documents.
            """
            return retriever.query(query, top_k=5)

        @tool
        def list_documents() -> str:
            """List all documents available in the current session."""
            docs = retriever.get_documents()
            if not docs:
                return "No documents uploaded yet."
            
            doc_list = [f"- {d.get('name', 'Unknown')}" for d in docs]
            return "Available documents:\n" + "\n".join(doc_list)

        return [retrieve_context, list_documents]

    def ask(self, question: str) -> str:
        """Ask the agent a question and return response text."""
        try:
            result = self.agent_executor.invoke({"input": question})
            return str(result.get("output", "")).strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Agent error: {e}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)}"

    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """Ask a question and return the answer with sources.
        Note: This bypasses memory for the direct source retrieval part but uses the agent for final synthesis.
        """
        # Get raw sources first
        sources = self.retriever.query_with_sources(question, top_k=5)
        
        # We can still use the agent but we'll guide it
        context_text = ""
        for i, s in enumerate(sources, 1):
            meta = s.get('metadata', {})
            cite = f"Source {i}"
            if meta.get('page_number'): cite += f", Page {meta['page_number']}"
            if meta.get('document_name'): cite += f" from {meta['document_name']}"
            context_text += f"\n--- {cite} ---\n{s['text']}\n"

        # Ask the agent with this extra context injected into the prompt
        enhanced_question = f"CONTEXT FROM DOCUMENTS:\n{context_text}\n\nQUESTION: {question}\n\nPlease answer based on the context above."
        
        answer = self.ask(enhanced_question)
        
        return {
            "answer": answer,
            "sources": sources
        }
