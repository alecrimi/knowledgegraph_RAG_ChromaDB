import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from knowledge_graph import KnowledgeGraphRAG

class MistralRAGSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Mistral API key from environment variable
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY must be set in .env file")
        
        # Initialize Mistral API client
        self.client = MistralClient(api_key=self.api_key)
        
        # Default model (can be changed)
        self.model = "mistral-large-latest"
        
        # Initialize Knowledge Graph
        self.knowledge_graph = KnowledgeGraphRAG()

    def augment_query(self, query: str) -> str:
        """
        Augment the query with relevant context from the knowledge graph
        
        Args:
            query (str): Original user query
        
        Returns:
            str: Augmented query with additional context
        """
        # Retrieve similar nodes
        similar_nodes = self.knowledge_graph.retrieve_similar_nodes(query)
        
        # Construct augmented context
        context = "\n".join([
            doc for doc in similar_nodes.get('documents', [])
        ])
        
        # Create a structured prompt with context
        augmented_prompt = f"""
        Context Information:
        {context}

        Based on the provided context and your extensive knowledge, 
        please answer the following query comprehensively:

        Query: {query}

        Your response should:
        1. Incorporate relevant information from the context
        2. Provide a detailed and informative answer
        3. Cite the context when directly relevant
        """
        
        return augmented_prompt

    def generate_response(self, augmented_query: str) -> str:
        """
        Generate response using Mistral Online API
        
        Args:
            augmented_query (str): Augmented query with context
        
        Returns:
            str: Generated response
        """
        try:
            # Prepare chat messages
            messages = [
                ChatMessage(role="system", content="You are a helpful AI assistant that uses provided context to generate comprehensive answers."),
                ChatMessage(role="user", content=augmented_query)
            ]
            
            # Generate response using Mistral API
            chat_response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=0.7,  # Creativity level
                max_tokens=500   # Maximum response length
            )
            
            # Extract and return the response
            return chat_response.choices[0].message.content
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def add_knowledge(self, node_id: str, content: str, metadata: dict = None):
        """
        Add knowledge to the graph
        
        Args:
            node_id (str): Unique node identifier
            content (str): Node content
            metadata (dict, optional): Additional metadata
        """
        self.knowledge_graph.add_node(node_id, content, metadata)

    def list_available_models(self):
        """
        List available Mistral models
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.list_models()
            return [model.id for model in models]
        except Exception as e:
            return [f"Error retrieving models: {str(e)}"]

    def change_model(self, model_name: str):
        """
        Change the current Mistral model
        
        Args:
            model_name (str): Name of the model to use
        """
        available_models = self.list_available_models()
        if model_name in available_models:
            self.model = model_name
            return f"Model changed to {model_name}"
        else:
            return f"Model {model_name} not available. Choose from: {available_models}"
