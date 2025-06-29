import os
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ====== Gemini API Configuration ======
GEMINI_API_KEY = "AIzaSyANW8MjQcmUOEK2HWtHxLYLqD3SsNUcfkQ"  
genai.configure(api_key=GEMINI_API_KEY)

# ====== Streamlit UI Setup ======
st.set_page_config(page_title="📘AI-Powered Document Q&A System", layout="centered")

# ====== Custom CSS for Background and Style ======
st.markdown(
    """
    <style>
        body {
            background-image: url("https://tse2.mm.bing.net/th/id/OIP.qAffK9huXFNQAo0g9FyFOwAAAA?w=296&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7");
        }
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #dff0ff 100%);
            padding: 20px;
            border-radius: 15px;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #333333;
            text-align: center;
            padding-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #666666;
            font-size: 18px;
            padding-bottom: 20px;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stTextInput>div>input {
            border: 1px solid #cccccc;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🤖 Gemini AI PDF Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload your PDF and ask anything from it!</div>", unsafe_allow_html=True)

# ====== PDF Upload & Query Input ======
uploaded_pdf = st.file_uploader("📎 Upload a PDF file", type="pdf")
query = st.text_input("💬 Ask a question based on the PDF content:")

# ====== Step 1: Extract PDF Text ======
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# ====== Step 2: Split and Embed ======
def embed_text_chunks(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return chunks, embeddings, model

# ====== Step 3: Create FAISS Index ======
def create_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# ====== Step 4: Retrieve Relevant Chunks ======
def retrieve_chunks(query, model, chunks, index, k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# ====== Step 5: Ask Gemini ======
def ask_gemini(prompt):
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")  # ✅ Stable version
    response = model.generate_content(prompt)
    return response.text

# ====== Main App Logic ======
if uploaded_pdf and query:
    with st.spinner("🔍 Processing your PDF and generating an answer..."):
        text = extract_text_from_pdf(uploaded_pdf)
        chunks, embeddings, embedding_model = embed_text_chunks(text)
        index = create_faiss_index(embeddings)
        top_chunks = retrieve_chunks(query, embedding_model, chunks, index)

        context = "\n\n".join(top_chunks)
        prompt = f"""
You are a helpful assistant. Answer the question based only on the following context:

--- Context Start ---
{context}
--- Context End ---

Question: {query}
Answer:
"""
        answer = ask_gemini(prompt)

    st.markdown("### ✅ Answer", unsafe_allow_html=True)
    st.success(answer)
