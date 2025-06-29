# 🤖 AI-Powered Document Q&A System

A Streamlit-based intelligent assistant that allows users to upload PDF documents and ask questions based on the content. It uses Retrieval-Augmented Generation (RAG) with Google Gemini, FAISS for similarity search, and Sentence Transformers for embedding.

## 📌 Features

- 📎 Upload any PDF document
- 🤖 Ask natural language questions based on the document content
- 🧠 Uses `sentence-transformers` for semantic embedding
- 🔍 Retrieves the most relevant document chunks using `FAISS`
- 💬 Uses `Gemini 1.5 Flash` to generate context-aware answers
- 🎨 Simple and modern Streamlit UI with custom background


## 🧠 Technologies Used
Streamlit: Used to build the web-based user interface for uploading PDFs and interacting with the chatbot.

PyMuPDF (fitz): Extracts raw text from uploaded PDF files.

FAISS: Performs efficient similarity search for finding relevant document chunks.

Sentence-Transformers: Converts document chunks and queries into vector embeddings.

LangChain: Handles intelligent text chunking to prepare data for embedding and retrieval.

Google Generative AI: Powers the Gemini 1.5 Flash LLM to generate accurate, context-aware answers based on retrieved text.


## 📁 Project Structure

