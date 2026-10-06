# 🎓 RAG-Based Personal Tutor — Small Prototype

A beginner-friendly RAG application that lets a student upload a PDF, ask questions about it, retrieve relevant passages, and generate grounded answers using a local Ollama LLM.

## Features

- PDF upload
- PDF text extraction
- Text chunking with overlap
- Sentence Transformer embeddings
- Semantic similarity retrieval
- Local LLM using Ollama
- Page-based source references
- AI document summary
- Streamlit web UI

## 1. Install Python

Use Python 3.10 or 3.11.

Check:

```bash
python --version
```

## 2. Create the project

```bash
mkdir rag-personal-tutor
cd rag-personal-tutor
```

Put the supplied files into this folder.

## 3. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may take some time because the embedding model is downloaded.

## 5. Install Ollama

Install Ollama from its official website:

https://ollama.com/

Then open Command Prompt / PowerShell and download the model:

```bash
ollama pull llama3.2:3b
```

Test it:

```bash
ollama run llama3.2:3b
```

Press Ctrl+C after confirming it works.

Ollama normally runs its local API at:

```text
http://localhost:11434
```

## 6. Run the application

From the project folder:

```bash
streamlit run app.py
```

The browser should open automatically.

If it does not, open:

```text
http://localhost:8501
```

## 7. How to use

1. Upload a PDF.
2. Wait for document processing.
3. Ask a question in the AI Tutor tab.
4. The application retrieves the most relevant document chunks.
5. The retrieved context is sent to the local LLM.
6. The answer is displayed with source page numbers.
7. Use Generate Summary for a quick study summary.

## RAG pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Sentence Transformer
 ↓
Embeddings
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
Ollama LLM
 ↓
Grounded Answer + Sources
```

## Project structure

```text
rag-personal-tutor/
│
├── app.py
├── requirements.txt
└── README.md
```

## Troubleshooting

### Ollama connection error

Make sure Ollama is running. Then test:

```bash
ollama list
```

You should see:

```text
llama3.2:3b
```

If missing:

```bash
ollama pull llama3.2:3b
```

### PowerShell virtual environment problem

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

### PDF has no extracted text

This prototype works best with text-based PDFs. Scanned/image-only PDFs need OCR, which can be added in the next version.

## Next upgrades

For a full MCA project, upgrade this prototype with:

- React frontend
- FastAPI backend
- PostgreSQL
- Qdrant/FAISS vector database
- Hybrid search
- Reranking
- User authentication
- Multiple documents/subjects
- MCQ generation
- Flashcards
- Exam preparation
- Learning analytics
- OCR
- Docker
- RAG evaluation
