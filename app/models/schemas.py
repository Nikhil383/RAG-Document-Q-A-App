from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    session_id: str
    question: str
    include_sources: bool = True
    streaming: bool = False

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    
class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks: int
    
class EvaluationRequest(BaseModel):
    session_id: str
    questions: Optional[List[str]] = None

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
