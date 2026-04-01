from langgraph.graph import StateGraph
from modules.llm.llm import get_llm


class GraphState(dict):
    """State for the RAG agent graph"""
    pass


def retrieve_node(state):
    """Retrieve relevant documents from the hybrid retriever"""
    retriever = state.get("retriever")
    query = state.get("query", "")
    
    if not retriever:
        return {"docs": [], "error": "No retriever configured"}
    
    docs = retriever.retrieve(query)
    return {"docs": docs, "query": query}


def generate_node(state):
    """Generate answer using LLM with retrieved context"""
    llm = get_llm()
    
    docs = state.get("docs", [])
    query = state.get("query", "")
    
    if not docs:
        return {
            "answer": "I couldn't find any relevant information to answer your question. Please try uploading some documents first.",
            "query": query
        }
    
    context = "\n\n".join([f"[Context {i+1}]:\n{doc}" for i, doc in enumerate(docs)])
    
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so honestly."""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return {"answer": answer, "query": query}


def build_graph(retriever):
    """Build and compile the RAG agent graph"""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # Set edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    
    # Compile with retriever in config
    return graph.compile()
