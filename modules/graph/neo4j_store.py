from neo4j import GraphDatabase

class Neo4jStore:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, cypher, params={}):
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [r.data() for r in result]