class GraphRetriever:
    def __init__(self, graph_store):
        self.graph = graph_store

    def search(self, query):
        cypher = """
        MATCH (c:Chunk)
        WHERE c.text CONTAINS $query
        RETURN c.text AS text
        LIMIT 5
        """
        results = self.graph.query(cypher, {"query": query})
        return [r["text"] for r in results]