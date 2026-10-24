# PDF Question-Answering App Using RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions about its contents.

The application extracts text from the PDF, splits it into smaller chunks, generates embeddings using Hugging Face, retrieves relevant chunks using FAISS, and generates answers using Google Gemini. A Streamlit interface allows users to interact with the RAG pipeline through a chat interface.

## Technologies Used

- Python
- Streamlit
- LangChain
- PyPDF
- Hugging Face Embeddings
- FAISS
- Google Gemini

## Project Structure

```text
Building_rag_pipeline/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .gitignore
└── data/
    └── genai_engineer_roadmap.pdf
```

## Project Files

- **`app.py`** — Implements the Streamlit interface, PDF upload and processing, chat interaction, streamed answers, conversation history, and retrieved-source display.
- **`rag_pipeline.py`** — Handles PDF text extraction, text chunking, embedding generation, FAISS indexing, similarity-based retrieval, and answer generation using Gemini.
- **`requirements.txt`** — Lists the Python packages required to run the application.
- **`.gitignore`** — Excludes local environment files and other unnecessary files from version control.
- **`data/genai_engineer_roadmap.pdf`** — Sample PDF used for standalone testing of the RAG pipeline.

## How It Works

1. The user uploads and processes a PDF through Streamlit.
2. The RAG pipeline extracts text, creates chunks and embeddings, and builds a FAISS index.
3. The user submits a question through the chat interface.
4. FAISS retrieves the most relevant chunks for that question.
5. Gemini generates an answer using the retrieved context.
6. Streamlit displays the answer progressively, along with source-page numbers and expandable retrieved text.
7. Conversation history is maintained.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chandu-123-git/Building_rag_pipeline.git
cd Building_rag_pipeline
```

### 2. Create and Activate a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root and add your Gemini API key:

```text
GOOGLE_API_KEY=your_actual_gemini_api_key
```

Replace the placeholder with your own API key. Keep the key private and never commit the `.env` file to GitHub.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

