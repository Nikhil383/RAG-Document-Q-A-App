import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.retrieval.reranker import reranking_layer
from app.retrieval.query_rewriter import query_rewriter
from app.db.redis_cache import redis_client
from app.models.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)

def _get_llm():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=settings.GEMINI_API_KEY
    )

class ChatService:
    def __init__(self):
        self.llm = _get_llm()
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the user's question based strictly on the provided context below. 
            If you don't know the answer, say "I don't know". Do not make up information.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )

    async def chat(self, session_id: str, question: str, include_sources: bool = True) -> ChatResponse:
        cache_key = f"{session_id}:q:{question}"
        cached_response = await redis_client.get(cache_key)
        if cached_response:
            logger.info("Serving from cache.")
            return ChatResponse(**cached_response)

        # 1. Rewrite Query
        rewritten_question = await query_rewriter.rewrite(question)
        logger.info(f"Rewritten Query: {rewritten_question}")

        # 2. Reranked Retrieval
        retriever = reranking_layer.get_reranked_retriever()
        docs = retriever.invoke(rewritten_question)
        
        # 3. Context Assembly
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 4. LLM Generation
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        answer = await chain.ainvoke({"context": context, "question": question})

        # 5. Populate Sources
        sources = []
        if include_sources:
            for d in docs:
                score = getattr(d.metadata, "relevance_score", 0.0) # BGE adds score usually
                sources.append(SourceDocument(
                    content=d.page_content,
                    metadata=d.metadata,
                    score=score
                ))

        response = ChatResponse(answer=answer, sources=sources)
        await redis_client.set(cache_key, response.model_dump())
        return response

chat_service = ChatService()
