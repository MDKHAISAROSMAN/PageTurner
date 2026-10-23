# PageTurner — Multimodal PDF Question Answering System

PageTurner is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about PDF documents and receive context-aware answers grounded in the document content. It combines document processing, semantic search, and local LLM inference to retrieve relevant information and generate responses.

The pipeline also incorporates summarization of tables and images to make visual document content more accessible during retrieval and answer generation.

## Features

- **PDF Processing:** Extracts content from PDF documents for downstream processing.
- **Title-Based Chunking:** Splits document content into chunks using document titles or section headings to preserve context.
- **Vector Search:** Generates embeddings and stores document chunks in ChromaDB for semantic retrieval.
- **Multimodal Content Summarization:** Incorporates summaries of tables and images into the document context.
- **Context-Aware Answers:** Uses retrieved document chunks to generate answers relevant to user queries.
- **Local LLM Inference:** Uses Ollama to run supported language models locally.
- **Response Evaluation:** Supports qualitative evaluation of answer accuracy, contextual relevance, and faithfulness to retrieved evidence.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | Document processing and RAG orchestration |
| Unstructured | PDF content extraction |
| ChromaDB | Vector storage and similarity search |
| Ollama | Local LLM inference and embeddings |
| Embedding Models | Convert document chunks into vector representations |

## How It Works

1. **Load PDF:** The application accepts a PDF document and extracts its content.
2. **Process Content:** Document text is organized into meaningful chunks using title-based chunking.
3. **Summarize Visual Content:** Tables and images are summarized and incorporated into the document context.
4. **Generate Embeddings:** Text chunks are converted into vector embeddings.
5. **Store in ChromaDB:** Embeddings and associated document content are stored in a vector database.
6. **Retrieve Relevant Chunks:** A user query is used to find semantically relevant document chunks.
7. **Generate an Answer:** Retrieved context is passed to an LLM, which generates an answer based on the available evidence.

## Installation

### Prerequisites

- Python 3.11 or another version compatible with your dependencies
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd PageTurner
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

If your project has a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies listed in your project configuration.

### 4. Configure Ollama

Start Ollama and download the models used by your application. For example:

```bash
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text
```

Use the exact model names configured in your code. The example above assumes your pipeline uses Qwen2.5-VL for answer generation and Nomic Embed Text for embeddings.

## Usage

1. Start Ollama and ensure the required models are available.
2. Run the application entry point:

```bash
python main.py
```

3. Provide a PDF file when prompted.
4. Enter questions about the document.
5. Review the generated answers and, where available, their supporting context.

The application is intended to answer questions using information retrieved from the supplied PDF rather than relying solely on the LLM's pretrained knowledge.

## Example

**Input query:**

> How many attention heads does the Transformer use, and what is the dimension of each head?

**Example response:**

> The Transformer uses 8 attention heads, with each head having a key and value dimension of 64, as described in *Attention Is All You Need*.

The answer should be checked against the retrieved document context to verify its accuracy.

## Project Structure

The following is an illustrative structure; adjust it to match your repository.

```text
PageTurner/
├── main.py
├── requirements.txt
├── README.md
├── ret_results.json
├── db/
│   └── chroma_db/
└── ...
```

## Limitations

- Answer quality depends on the quality of PDF extraction, chunking, embeddings, and retrieval.
- Incorrect or incomplete visual summaries may affect answers involving tables and images.
- The model may produce incorrect conclusions even when relevant context is retrieved.
- Local inference performance depends on the selected model and available hardware.
- The application should be evaluated on questions that are answerable from the document as well as questions for which the document contains insufficient information.

## Future Improvements

- Improve retrieval through hybrid search and reranking.
- Add source citations for generated answers.
- Improve table and image extraction and summarization.
- Introduce automated evaluation for retrieval relevance and answer faithfulness.
- Add a user interface for uploading PDFs and interacting with documents.

---

**PageTurner** — Retrieve relevant context. Generate grounded answers.
