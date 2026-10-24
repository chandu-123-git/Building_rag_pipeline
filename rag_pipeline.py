
import os
import tempfile

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# Load environment variables from .env
load_dotenv()


# Load the embedding model once
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)


# Prepare the prompt
prompt = ChatPromptTemplate.from_template("""
You are a document question-answering assistant.

Answer using only the provided context.
If the context does not contain enough information, respond:
"I couldn't find the answer in the provided document."

Do not invent facts or use outside knowledge.
Keep the answer clear and concise.

Context:
{context}

Question:
{question}
""")


def process_pdf(pdf_bytes):
    """Extract PDF text, create chunks, and build a FAISS index."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    # Save the uploaded PDF temporarily for PyPDFLoader
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
    ) as temp_file:
        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    try:
        # Load the PDF one page at a time
        loader = PyPDFLoader(pdf_path, mode="page")
        pages = loader.load()
    finally:
        # Remove the temporary file after loading
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    # Check whether the PDF contains extractable text
    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "No extractable text was found in this PDF. "
            "Scanned PDFs may require OCR."
        )

    # Split each page into smaller text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    # Generate an embedding for every chunk
    vectors = embeddings_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    # Convert embeddings into 32-bit floating-point values
    document_embeddings = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    # Normalize vectors for cosine similarity search
    faiss.normalize_L2(document_embeddings)

    # Create a FAISS index using inner product
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Store all chunk embeddings in the index
    index.add(document_embeddings)

    # Keep this document's chunks and index together
    return {
        "chunks": chunks,
        "index": index,
    }


def retrieve_data(question, document_data):
    """Retrieve the three most relevant chunks for a question."""

    chunks = document_data["chunks"]
    index = document_data["index"]

    # Convert the question into an embedding
    query_vector = embeddings_model.embed_query(question)

    query_embedding = np.ascontiguousarray(
        [query_vector],
        dtype=np.float32,
    )

    # Normalize the query for cosine similarity search
    faiss.normalize_L2(query_embedding)

    # Retrieve up to three relevant chunks
    top_k = min(3, index.ntotal)

    if top_k == 0:
        return {
            "question": question,
            "context": "",
            "source_pages": [],
        }

    _, ids = index.search(query_embedding, k=top_k)

    retrieved_texts = []
    source_pages = []

    for chunk_id in ids[0]:
        if chunk_id == -1:
            continue

        chunk = chunks[int(chunk_id)]
        page = chunk.metadata.get("page")

        if page is not None:
            # PyPDFLoader page numbers start at zero
            page_number = page + 1

            if page_number not in source_pages:
                source_pages.append(page_number)

            retrieved_texts.append(
                f"[Source page {page_number}]\n"
                f"{chunk.page_content}"
            )
        else:
            retrieved_texts.append(chunk.page_content)

    return {
        "question": question,
        "context": "\n\n".join(retrieved_texts),
        "source_pages": source_pages,
    }


def create_rag_chain(document_data):
    """Create a RAG chain for the selected document."""

    return (
        RunnableLambda(
            lambda question: retrieve_data(question, document_data)
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )


# Run a standalone test when this file is executed directly
if __name__ == "__main__":
    test_pdf_path = "data/genai_engineer_roadmap.pdf"

    with open(test_pdf_path, "rb") as pdf_file:
        document_data = process_pdf(pdf_file.read())

    rag_chain = create_rag_chain(document_data)

    query = "What topics are covered in the RAG learning step?"
    result = rag_chain.invoke(query)

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Source Pages:")
    if result["source_pages"]:
        print(", ".join(map(str, result["source_pages"])))
    else:
        print("Source page information is unavailable.")
