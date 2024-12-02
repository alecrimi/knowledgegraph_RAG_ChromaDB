import chainlit as cl
from rag_implementation import MistralRAGSystem

# Initialize RAG system
rag_system = MistralRAGSystem()

# Pre-populate knowledge graph with some initial data
def initialize_knowledge_base():
    knowledge_items = [
        {
            "id": "ai_basics",
            "content": "Artificial Intelligence is a broad field of computer science focused on creating intelligent machines that can simulate human-like thinking and learning capabilities.",
            "metadata": {"category": "introduction", "difficulty": "beginner"}
        },
        {
            "id": "ml_fundamentals",
            "content": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed, using algorithms that can learn from and make predictions or decisions based on data.",
            "metadata": {"category": "core_concept", "difficulty": "intermediate"}
        }
    ]
    
    for item in knowledge_items:
        rag_system.add_knowledge(item["id"], item["content"], item["metadata"])

# Initialize knowledge base
initialize_knowledge_base()

@cl.on_chat_start
async def start():
    await cl.Message(content="RAG System with Mistral is ready! How can I help you today?").send()

@cl.on_message
async def main(message: str):
    # Augment the query with relevant context
    augmented_query = rag_system.augment_query(message)
    
    # Generate response
    response = rag_system.generate_response(augmented_query)
    
    # Send the response back to the user
    await cl.Message(content=response).send()

# Optional: Add a way to dynamically add knowledge
@cl.on_message(pattern="^/add_knowledge")
async def add_knowledge(message: str):
    # Parse the message to extract node_id and content
    parts = message.split(maxsplit=3)
    if len(parts) < 3:
        await cl.Message(content="Usage: /add_knowledge <node_id> <content>").send()
        return
    
    node_id, content = parts[1], parts[2]
    rag_system.add_knowledge(node_id, content)
    await cl.Message(content=f"Added knowledge node: {node_id}").send()

