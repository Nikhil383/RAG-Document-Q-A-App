import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

def _get_gemini_models():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
    return llm, embeddings

class RagasEvaluator:
    @staticmethod
    def evaluate_results(questions: list[str], contexts: list[list[str]], answers: list[str], ground_truths: list[list[str]] = None):
        """
        Evaluate generated answers against Ragas metrics using Gemini.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths is not None:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        
        llm, embeddings = _get_gemini_models()
        
        # Override the models with Gemini
        metrics = [faithfulness, answer_relevance, context_precision]
        if ground_truths is not None:
            metrics.append(context_recall)

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        return result
