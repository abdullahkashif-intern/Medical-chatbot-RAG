# Medical Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Groq LLM, and Pinecone vector database for medical Q&A. The system retrieves relevant medical information from PDF documents and generates accurate answers using a large language model.

## Features

- **Document Ingestion**: Load and process medical PDFs from local storage
- **Text Chunking**: Split documents into manageable chunks (500 tokens, 20-token overlap) for optimal retrieval
- **Vector Embeddings**: Generate embeddings using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model (384-dim vectors)
- **Vector Storage**: Store and retrieve embeddings using Pinecone serverless infrastructure
- **RAG Pipeline**: Combine retrieved context with LLM for grounded, context-aware answers
- **LLM Integration**: Groq's `llama-3.3-70b-versatile` model for fast, high-quality responses
- **Modular Architecture**: Separated concerns across helper functions, prompts, and main pipeline

## Architecture

```
PDFs → Load & Process → Text Splitting → Embeddings → Pinecone Index
                                                            ↓
                                                        Retriever
                                                            ↓
User Query → Embed → Retrieve Top-K → Groq LLM + Prompt → Response
```

## Installation

### Prerequisites
- Python 3.12+
- API Keys: Groq, HuggingFace, Pinecone, Google (optional)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdullahkashif-intern/Medical-chatbot-RAG.git
   cd Medical-chatbot-RAG
   ```

   ```

3. **Install dependencies**
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── helper.py           # Document loading, chunking, embeddings
│   └── prompt.py           # System prompt for QA tasks
├── research/
│   └── trials.ipynb        # Experimentation notebook
├── data/
│   └── Medical_book.pdf    # Sample medical reference
├── index_starter.py        # Main RAG pipeline orchestration
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Pip requirements (legacy)
├── uv.lock                 # Locked dependency versions
└── README.md               # This file
```

## Usage

### Quick Start

Run the complete RAG pipeline:
```bash
python index_starter.py
```

This will:
1. Load embeddings from HuggingFace
2. Connect to Pinecone (create index if needed)
3. Load & process PDFs from `data/` directory
4. Split documents into chunks
5. Generate and store embeddings
6. Build RAG chain with Groq LLM
7. Answer: "What is Acne?"
## Configuration

### Pinecone Index
- **Dimension**: 384 (matches HuggingFace embedding size)
- **Metric**: Cosine similarity
- **Spec**: Serverless (AWS, us-east-1)
- **Index Name**: `medical-chatbot`

### Groq LLM
- **Model**: `llama-3.3-70b-versatile`
- **Temperature**: 0.4 (balanced: creative but accurate)
- **Max Tokens**: Default (auto)

### Text Splitting
- **Chunk Size**: 500 tokens
- **Overlap**: 20 tokens (context continuity)
- **Splitter**: RecursiveCharacterTextSplitter

### Retrieval
- **Search Type**: Similarity
- **Top-K Results**: 3 documents per query
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

## Modular Components

### `src/helper.py`
**Purpose**: Document processing utilities

**Functions**:
- `load_pdf_files(data_dir)` - Load PDFs using DirectoryLoader
- `filter_to_minimal_docs(docs)` - Extract source and content, strip metadata noise
- `text_split(docs)` - Split into chunks with overlap
- `download_hugging_face_embeddings()` - Get embedding model from HF Inference API

### `src/prompt.py`
**Purpose**: RAG system prompt template

**Content**: Instructs LLM to:
- Use retrieved context for answers

### `index_starter.py`
**Purpose**: Main orchestration script

**Functions**:
- `build_docsearch(embeddings)` - Create/reuse Pinecone index, populate from PDFs
- `build_rag_chain(docsearch)` - Assemble retriever, LLM, and chain
- `main()` - Entry point; runs end-to-end pipeline

## Workflow

1. **Initialization**: Load HuggingFace embeddings (384-dim vectors)
2. **Index Management**: Check if Pinecone index exists
   - If yes: Reuse existing index
   - If no: Create, load PDFs, chunk, embed, and upsert
3. **Retrieval Setup**: Create similarity-based retriever (k=3)
4. **Chain Assembly**: Combine retriever + Groq LLM + prompt template
5. **Query**: Send user question → retrieve top-3 chunks → feed to LLM → response

## Performance Notes

- **Embedding**: ~50-100ms per chunk (HF Inference API)
- **Retrieval**: ~100-200ms (Pinecone)
- **Generation**: ~500-2000ms (Groq, depends on output length)
- **Total**: ~1-3 seconds per query

## Future Enhancements

- [ ] Streaming responses for long answers
- [ ] Multi-turn conversation memory
- [ ] Hybrid search (dense + sparse retrieval)
- [ ] Query expansion / reranking
- [ ] Fine-tuned medical embeddings
- [ ] FastAPI REST endpoint
- [ ] Web UI (Streamlit/Gradio)

## Authors

- **Abdullah Kashif** 
