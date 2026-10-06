import os
import streamlit as st
import fitz
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Personal Tutor", page_icon="🎓", layout="wide")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_pages(uploaded_file):
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for p in pages:
        text = " ".join(p["text"].split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "page": p["page"]})
            if end >= len(text):
                break
            start = end - overlap
    return chunks

def retrieve(query, chunks, embeddings, model, top_k=4):
    q = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q, embeddings)[0]
    ids = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in ids]

def ask_ollama(question, contexts):
    context = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in contexts
    )
    prompt = f"""You are a helpful personal tutor.
Answer the student's question ONLY using the provided document context.
If the answer is not present in the context, clearly say:
"I couldn't find that information in the uploaded document."
Do not invent facts. Give a clear student-friendly explanation.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return (
            "Ollama could not be reached. Make sure Ollama is running and the "
            f"model `{MODEL_NAME}` is installed.\n\nError: " + str(e)
        )

def simple_summary(chunks, model):
    # Prototype summary: retrieve representative chunks and ask the local LLM.
    sample = chunks[: min(8, len(chunks))]
    return ask_ollama(
        "Create a concise study summary with key concepts and important definitions.",
        sample
    )

st.title("🎓 RAG-Based Personal Tutor")
st.caption("Small working prototype: upload a PDF and ask questions grounded in it.")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.filename = None
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 Study Material")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        if st.session_state.filename != uploaded.name:
            with st.spinner("Processing document..."):
                pages = extract_pages(uploaded)
                chunks = make_chunks(pages)
                model = load_embedding_model()
                embeddings = model.encode(
                    [c["text"] for c in chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.filename = uploaded.name
                st.session_state.messages = []
            st.success(f"Processed {len(chunks)} chunks.")

    if st.session_state.chunks:
        st.divider()
        st.write(f"**Document:** {st.session_state.filename}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")

        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = simple_summary(st.session_state.chunks, load_embedding_model())
            st.session_state.summary = summary

if not st.session_state.chunks:
    st.info("👈 Upload a PDF from the sidebar to start.")
    st.markdown("""
    ### What this prototype demonstrates
    - PDF text extraction
    - Text chunking
    - Sentence-transformer embeddings
    - Semantic retrieval
    - Local LLM generation with Ollama
    - Page-based source references
    - Document summarization
    """)
else:
    tab1, tab2 = st.tabs(["💬 AI Tutor", "📝 Summary"])

    with tab1:
        st.subheader(f"Ask questions about **{st.session_state.filename}**")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask something from your document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            model = load_embedding_model()
            contexts = retrieve(
                question,
                st.session_state.chunks,
                st.session_state.embeddings,
                model
            )

            with st.chat_message("assistant"):
                with st.spinner("Searching your document..."):
                    answer = ask_ollama(question, contexts)
                st.markdown(answer)

                st.markdown("#### 📚 Sources")
                for c in contexts:
                    st.caption(
                        f"Page {c['page']} · relevance {c['score']:.2f}"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

    with tab2:
        st.subheader("📝 AI Study Summary")
        if "summary" in st.session_state:
            st.markdown(st.session_state.summary)
        else:
            st.info("Click **Generate Summary** in the sidebar.")
