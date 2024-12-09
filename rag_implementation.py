import os
from dotenv import load_dotenv
import requests
from graph_embedding import KnowledgeGraphRAG

class MistralRAGSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Hugging Face API key from environment variable
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY must be set in .env file")
        
        # Default model (corrected name)
        self.model = "mistralai/Mistral-7B-v0.1"  
                
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
        
        # If similar_nodes is a list, iterate over it directly
        context = "\n".join([str(doc) for doc in similar_nodes])
        
        # Create a structured prompt with context
        augmented_prompt = f"""
        #Context Information:
        #{context}

        Based on the provided context and your extensive knowledge, 
        please answer the following query comprehensively:

        Query: {query}

        Response:
        """
       
        return augmented_prompt
    '''
    # Your response should:
        1. Incorporate relevant information from the context
        2. Provide a detailed and informative answer
        3. Cite the context when directly relevant
        4. Only when the information is not in the provided context, use the main language model
    '''

    def generate_response(self, augmented_query: str) -> str:
        """
        Generate response using Hugging Face API for Mistral model
        
        Args:
            augmented_query (str): Augmented query with context
        
        Returns:
            str: Generated response
        """
        try:
            # Prepare headers with the Hugging Face API key
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare payload
            payload = {
                'inputs': augmented_query
            }

            # Hugging Face Inference API endpoint for Mistral model
            url = f'https://api-inference.huggingface.co/models/{self.model}'

            # Make the POST request to generate a response
            response = requests.post(url, json=payload, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                #return response.json()[0]['generated_text']
                generated_text = response.json()[0]['generated_text']
                
                
                print("Raw response:", response.json())
                
                start_index = generated_text.find("Response:") + len("Response:")
                response_without_context = generated_text[start_index:].strip()
                
                #response_without_context=generated_text
                
                # Remove any remaining instructions or artifacts
                #response_without_context = response_without_context.split("\n", 1)[0]  # Take only the first logical sentence

                
                #print(response_without_context)
                return response_without_context
            else:
                return f"Error: {response.status_code} - {response.text}"

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
        List available Mistral models (Hugging Face)
        
        Returns:
            List of available model names
        """
        # Hugging Face doesn't provide a direct API for listing models
        return ["List of models available on Hugging Face Hub: Search at https://huggingface.co/models"]

    def change_model(self, model_name: str):
        """
        Change the current Mistral model
        
        Args:
            model_name (str): Name of the model to use
        """
        # For Hugging Face, we can't list models directly via the API
        self.model = model_name
        return f"Model changed to {model_name}"

