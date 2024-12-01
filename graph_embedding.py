import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from typing import List, Dict, Any

class KnowledgeGraphRAG:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="knowledge_base")

    def add_node(self, node_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a node to the knowledge graph and embed its content
        
        Args:
            node_id (str): Unique identifier for the node
            content (str): Text content of the node
            metadata (dict, optional): Additional metadata for the node
        """
        # Add to networkx graph
        self.graph.add_node(node_id, content=content, metadata=metadata or {})
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=[node_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def add_edge(self, source: str, target: str, relationship: str = None):
        """
        Add a directed edge between two nodes
        
        Args:
            source (str): Source node ID
            target (str): Target node ID
            relationship (str, optional): Type of relationship
        """
        self.graph.add_edge(source, target, relationship=relationship)

    def retrieve_similar_nodes(self, query: str, top_k: int = 3):
        """
        Retrieve most similar nodes to a given query
        
        Args:
            query (str): Search query
            top_k (int): Number of top similar nodes to retrieve
        
        Returns:
            List of most similar nodes
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results

    def visualize_graph(self, output_path: str = 'knowledge_graph.png'):
        """
        Visualize the knowledge graph
        
        Args:
            output_path (str): Path to save the graph visualization
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        plt.title("Knowledge Graph Visualization")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# Example usage
def create_sample_knowledge_graph():
    kg = KnowledgeGraphRAG()
    
    # Add some sample nodes about AI
    kg.add_node("ai_intro", "Artificial Intelligence is a branch of computer science")
    kg.add_node("ml_intro", "Machine Learning is a subset of AI focusing on learning from data")
    kg.add_node("dl_intro", "Deep Learning uses neural networks with multiple layers")
    
    # Add some relationships
    kg.add_edge("ai_intro", "ml_intro", "contains")
    kg.add_edge("ml_intro", "dl_intro", "advanced_technique")
    
    return kg

# For testing
if __name__ == "__main__":
    kg = create_sample_knowledge_graph()
    kg.visualize_graph()
    
    # Example retrieval
    results = kg.retrieve_similar_nodes("neural networks")
    print(results)

