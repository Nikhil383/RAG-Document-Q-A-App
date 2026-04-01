from neo4j import GraphDatabase

class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_graph(self, chunks):
        """Build knowledge graph from chunks"""
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                doc_id = chunk.metadata.get("doc_id", f"doc_{i}")
                page = chunk.metadata.get("page", 0)
                
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    MERGE (p:Page {number: $page, doc_id: $doc_id})
                    MERGE (d)-[:HAS_PAGE]->(p)

                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text,
                        c.doc_id = $doc_id,
                        c.page = $page

                    MERGE (p)-[:HAS_CHUNK]->(c)
                """, {
                    "doc_id": doc_id,
                    "page": page,
                    "chunk_id": f"{doc_id}_{chunk.metadata.get('chunk_idx', i)}",
                    "text": chunk.page_content
                })