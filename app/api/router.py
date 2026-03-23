import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Dict
from app.models.schemas import ChatRequest, ChatResponse, EvaluationRequest, EvaluationResponse, UploadResponse
from app.services.chat_service import chat_service
from app.services.document_service import DocumentService
from app.services.evaluation_service import EvaluationService
from app.core.config import settings

api_router = APIRouter()

@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await chat_service.chat(
            session_id=request.session_id,
            question=request.question,
            include_sources=request.include_sources
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.csv')):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")
        
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        result = DocumentService.process_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return result

@api_router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pipeline(request: EvaluationRequest):
    try:
        results = await EvaluationService.evaluate_knowledge_base(request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
