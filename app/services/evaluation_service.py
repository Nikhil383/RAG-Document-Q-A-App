import uuid
from typing import List, Optional
from app.evaluation.ragas_eval import RagasEvaluator
from app.services.chat_service import chat_service
from app.models.schemas import EvaluationResponse, EvaluationRequest
from app.retrieval.hybrid_search import hybrid_retriever

class EvaluationService:
    @staticmethod
    async def evaluate_knowledge_base(request: EvaluationRequest) -> EvaluationResponse:
        questions = request.questions
        if not questions:
            # Synthetic questions if none provided
            questions = [
                "What is the main topic of the uploaded documents?",
                "Can you summarize the key findings?"
            ]
            
        contexts_list = []
        answers_list = []
        
        # We need to run the RAG pipeline for each question to generate answers and contexts
        for q in questions:
            resp = await chat_service.chat(request.session_id, q, include_sources=True)
            answers_list.append(resp.answer)
            contexts = [src.content for src in resp.sources]
            contexts_list.append(contexts)
            
        # Run Ragas Evaluation
        result = RagasEvaluator.evaluate_results(
            questions=questions, 
            contexts=contexts_list, 
            answers=answers_list
        )
        
        results_dict = result.to_dict() if hasattr(result, 'to_dict') else result
        
        return EvaluationResponse(
            faithfulness=results_dict.get('faithfulness', 0.0),
            answer_relevance=results_dict.get('answer_relevance', 0.0),
            context_precision=results_dict.get('context_precision', 0.0),
            context_recall=results_dict.get('context_recall', 0.0),
            overall_score=sum([
                results_dict.get('faithfulness', 0.0),
                results_dict.get('answer_relevance', 0.0),
                results_dict.get('context_precision', 0.0)
            ]) / 3
        )
