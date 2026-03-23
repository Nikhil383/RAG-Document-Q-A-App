from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.retrieval.hybrid_search import hybrid_retriever

class RerankingLayer:
    def __init__(self):
        # We use BGE-based reranker to sort our hybrid search results
        self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.compressor = CrossEncoderReranker(model=self.model, top_n=3)

    def get_reranked_retriever(self):
        base_retriever = hybrid_retriever.get_retriever()
        return ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )

reranking_layer = RerankingLayer()
