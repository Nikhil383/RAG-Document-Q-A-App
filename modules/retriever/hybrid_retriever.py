class HybridRetriever:
    """Combines vector and graph retrieval for better results"""
    
    def __init__(self, vector_store, graph_retriever):
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
    
    def retrieve(self, query, k=5):
        """Retrieve from both sources and combine results"""
        vector_results = self.vector_store.search(query, k=k)
        graph_results = self.graph_retriever.search(query, k=k)
        
        # Combine and deduplicate results
        all_results = []
        seen_texts = set()
        
        for doc in vector_results:
            if doc.page_content not in seen_texts:
                all_results.append(doc.page_content)
                seen_texts.add(doc.page_content)
        
        for text in graph_results:
            if text not in seen_texts:
                all_results.append(text)
                seen_texts.add(text)
        
        return all_results
