from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

def _get_llm():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY
    )

class QueryRewriter:
    def __init__(self):
        self.llm = _get_llm()
        
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at optimizing search queries for a Retrieval-Augmented Generation system. Your task is to rewrite the given user query to be as clear and semantically rich as possible, extracting its core intent. Return only the rewritten query, nothing else."),
            ("human", "Original Query: {question}")
        ])
        
        self.chain = self.rewrite_prompt | self.llm | StrOutputParser()

    async def rewrite(self, question: str) -> str:
        # For simplicity, we directly rewrite
        rewritten = await self.chain.ainvoke({"question": question})
        return rewritten.strip()

query_rewriter = QueryRewriter()
