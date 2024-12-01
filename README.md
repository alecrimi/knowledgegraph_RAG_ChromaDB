# 🚀 Mistral RAG System with Knowledge Graph

## 📋 Overview

This project implements a cutting-edge Retrieval-Augmented Generation (RAG) system leveraging:

- 🤖 Mistral AI Large Language Model
- 🗃️ ChromaDB for Vector Storage
- 🔍 Sentence Transformers for Semantic Embeddings
- 💬 Chainlit for Interactive Interface
- 🌐 NetworkX for Knowledge Graph Visualization

## ✨ Features

- 📊 Dynamic Knowledge Graph Management
- 🔮 Semantic Search Capabilities
- 🧠 Context-Aware Response Generation
- 💻 Interactive Chat Interface
- 🌱 Real-time Knowledge Base Expansion

## 🛠️ Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Mistral AI API Key

## 🚀 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mistral-rag-system.git
cd mistral-rag-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# OR
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

## 🏃 Running the Application
```bash
chainlit run chainlit_app.py
```

## 📝 Usage Examples

### Adding Knowledge Dynamically
In the Chainlit interface, use:
```
/add_knowledge <node_id> <content>
```

## 🔬 Components

- `knowledge_graph.py`: NetworkX & ChromaDB Knowledge Graph
- `rag_implementation.py`: Mistral AI RAG Logic
- `chainlit_app.py`: Interactive Interface

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🌟 Acknowledgements

- [Mistral AI](https://mistral.ai)
- [Chainlit](https://chainlit.io)
- [ChromaDB](https://www.trychroma.com/)

---

**Note**: Always ensure you comply with Mistral AI's usage terms and conditions.
