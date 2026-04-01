class GraphRetriever:
    def __init__(self, graph_store):
        self.graph = graph_store

    def search(self, query, k=5):
        """Search graph for relevant chunks using keyword matching"""
        cypher = """
        MATCH (c:Chunk)
        WHERE c.text CONTAINS $query
        RETURN c.text AS text
        LIMIT $limit
        """
        results = self.graph.query(cypher, {"query": query, "limit": k})
        return [r["text"] for r in results]
