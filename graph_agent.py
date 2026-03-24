import os
import operator
from typing import TypedDict, Annotated, List, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. State Definition
class AgentState(TypedDict):
    query: str
    entities: List[str]
    context: str
    answer: str
    # 'operator.add' appends to the count recursively during the loop
    refine_count: Annotated[int, operator.add] 

class EntitiesOutput(BaseModel):
    entities: List[str] = Field(description="A list of core entities or nouns extracted from the user's prompt.")

# Helper to init connections
def initialize_components():
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    try:
        graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
    except Exception:
        # Fills in a dummy graph if Neo4j is not running on the host
        graph = None 
    return llm, graph

# 2. Nodes
def extract_entities(state: AgentState):
    """Uses LLM structured output to reliably extract entities from the prompt"""
    llm, _ = initialize_components()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the user's query and extract up to 3 core entities to search across a Neo4j Knowledge Graph."),
        ("human", "{query}")
    ])
    
    # Modern tool calling and structured output binding
    chain = prompt | llm.with_structured_output(EntitiesOutput)
    res = chain.invoke({"query": state["query"]})
    
    print(f"[Node: extract_entities] Extracted: {res.entities}")
    return {"entities": res.entities}

def retrieve_graph_context(state: AgentState):
    """Hits Neo4j using a Cypher Query dynamically built from the extracted entities"""
    print(f"[Node: retrieve_graph_context] Querying Neo4j for {state['entities']}...")
    _, graph = initialize_components()
    
    if not graph:
        return {"context": "Graph Database Unavailable."}
    
    context_str = ""
    for entity in state["entities"]:
        # Safe parameterized hybrid query for relationships 
        cypher = f"""
        MATCH (n)-[r]-(m) 
        WHERE toLower(n.id) CONTAINS toLower('{entity}')
        RETURN n.id AS source, type(r) AS relationship, m.id AS target
        LIMIT 10
        """
        try:
            results = graph.query(cypher)
            for row in results:
                context_str += f"{row['source']} -[{row['relationship']}]-> {row['target']}\n"
        except Exception as e:
            print(f"Cypher error: {e}")
            
    if not context_str.strip():
        context_str = "No specific relationships found in the knowledge graph."
        
    return {"context": context_str}

def generate_answer(state: AgentState):
    """Formulates a comprehensive answer strictly based on the extracted subgraph"""
    print(f"[Node: generate_answer] Analyzing context...")
    llm, _ = initialize_components()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Graph RAG Assistant. Use ONLY the provided Knowledge Graph relationships to answer the query.\n"
         "If the context is empty or entirely irrelevant, do NOT hallucinate an answer. Instead, reply EXACTLY with: 'INSUFFICIENT_CONTEXT'."
        ),
        ("human", "Graph Relationships:\n{context}\n\nQuery: {query}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({"context": state["context"], "query": state["query"]})
    
    answer = res.content.strip()
    return {"answer": answer, "refine_count": 1}

# 3. Conditional Edge Routing
def check_refinement(state: AgentState) -> str:
    """Decides dynamically whether to loop back to extraction or end the execution"""
    print(f"   [Edge Check] Answer generated. Refinement Count: {state['refine_count']}")
    
    if "INSUFFICIENT_CONTEXT" in state["answer"] and state["refine_count"] < 3:
        print("   [Routing] -> Insufficient context. Triggering graph search refinement loop...")
        return "refine"
        
    print("   [Routing] -> Search sufficient (or max iterations reached). Terminating process.")
    return "end"

# 4. Graph Architecture Assembly
def build_agent():
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("retrieve_graph_context", retrieve_graph_context)
    workflow.add_node("generate_answer", generate_answer)
    
    # Establish Standard Edges
    workflow.set_entry_point("extract_entities")
    workflow.add_edge("extract_entities", "retrieve_graph_context")
    workflow.add_edge("retrieve_graph_context", "generate_answer")
    
    # Establish Conditional Edges for Agentic refinement
    workflow.add_conditional_edges(
        "generate_answer",
        check_refinement,
        {
            "refine": "extract_entities", # Loops back to start
            "end": END
        }
    )
    
    return workflow.compile()

if __name__ == "__main__":
    # Test Execution Example
    agent = build_agent()
    
    # Requires a running Neo4j instance at bolt://localhost:7687
    inputs = {
        "query": "Who created LangChain and what is Neo4j?",
        "refine_count": 0
    }
    
    print("--- Starting Agentic Graph RAG Execution ---")
    result = agent.invoke(inputs)
    
    print("\n\n--- Final Result ---")
    print(result["answer"])
