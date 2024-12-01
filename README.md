# knowledgegraph_RAG_ChromaDB


RAG System with Mistral Online API, ChromaDB, and Chainlit
Setup Instructions
Prerequisites

Python 3.8+
Mistral AI API Key

Installation

Clone the repository
Create a virtual environment

bashCopypython -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies

bashCopypip install -r requirements.txt
API Key Configuration

Sign up for a Mistral AI API key at Mistral AI Platform
Create a .env file in the project root
Add your API key:

CopyMISTRAL_API_KEY=your_mistral_api_key_here
Running the Application
bashCopychainlit run chainlit_app.py
New Features

Direct Mistral Online API Integration
Dynamic Model Selection
Secure API Key Management
Fallback Error Handling

Usage

Interact via Chainlit interface
Use /add_knowledge <node_id> <content> to expand knowledge base
