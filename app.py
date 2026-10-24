
import hashlib

import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DocuMind | Chat with your PDF",
    page_icon="📄",
    layout="centered",
)


# --------------------------------------------------
# Styling: clean, chat-first interface
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    .brand-label {
        color: #0D9488;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-top: -0.4rem;
        margin-bottom: 1.5rem;
    }

    .document-status {
        color: #0F766E;
        font-weight: 600;
        font-size: 0.9rem;
    }

    div.stButton > button[kind="primary"] {
        background-color: #0D9488;
        border-color: #0D9488;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #0F766E;
        border-color: #0F766E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Reuse the existing RAG backend
# --------------------------------------------------
@st.cache_resource
def load_pipeline_functions():
    from rag_pipeline import process_pdf, create_rag_chain

    return process_pdf, create_rag_chain


process_pdf, create_rag_chain = load_pipeline_functions()


# --------------------------------------------------
# Conversation and document state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# App heading
# --------------------------------------------------
st.markdown(
    '<div class="brand-label">Document intelligence</div>',
    unsafe_allow_html=True,
)

st.title("DocuMind")
st.markdown(
    '<div class="subtitle">'
    'Chat with your PDF. Ask questions and explore the evidence '
    'behind each answer.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Attach and process a PDF
# --------------------------------------------------
with st.container(border=True):
    st.markdown("**📎 Attach a document**")
    st.caption(
        "Upload a PDF to prepare it for question answering. "
        "Your document is processed before the chat becomes available."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    selected_hash = None

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        selected_hash = hashlib.sha256(pdf_bytes).hexdigest()

        st.caption(
            f"Selected: **{uploaded_file.name}** · "
            f"{len(pdf_bytes) / 1024:.1f} KB"
        )

        is_already_processed = (
            selected_hash == st.session_state.document_hash
            and st.session_state.rag_chain is not None
        )

        if is_already_processed:
            st.success("This document is ready for questions.")

        elif st.button(
            "Process PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Preparing your document: extracting text, "
                    "creating embeddings, and building the FAISS index..."
                ):
                    document_data = process_pdf(pdf_bytes)
                    new_rag_chain = create_rag_chain(document_data)

                # Activate the document only after processing succeeds.
                st.session_state.rag_chain = new_rag_chain
                st.session_state.document_hash = selected_hash
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.pending_question = None

                st.rerun()

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")

    else:
        st.caption("No document attached yet.")


# --------------------------------------------------
# Enable chat only for the selected, processed PDF
# --------------------------------------------------
active_document = (
    uploaded_file is not None
    and selected_hash == st.session_state.document_hash
    and st.session_state.rag_chain is not None
)

if not active_document:
    st.info(
        "Attach a PDF and click **Process PDF** to start chatting."
    )

else:
    st.markdown(
        f'<div class="document-status">'
        f'● Ready to chat · {st.session_state.document_name}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Show a helpful starting state for a new conversation.
    if not st.session_state.messages:
        st.markdown("### What would you like to know?")
        st.caption(
            "Ask in your own words. Answers are generated using "
            "relevant passages retrieved from your document."
        )

        suggestions = [
            "Summarize the main topics.",
            "What are the key concepts?",
            "Explain the RAG learning step.",
        ]

        columns = st.columns(3)

        for i, suggestion in enumerate(suggestions):
            with columns[i]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = suggestion

    # Display the conversation so far.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                with st.expander("View retrieved sources"):
                    if message["source_pages"]:
                        st.write(
                            "**Source pages:** "
                            + ", ".join(
                                str(page)
                                for page in message["source_pages"]
                            )
                        )

                    if message["context"]:
                        st.markdown("**Retrieved text**")
                        st.text(message["context"])

    # Accept a typed question or a suggested question.
    typed_question = st.chat_input(
        "Ask a question about your PDF...",
    )

    question = typed_question

    if question is None and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        stream_state = {
            "source_pages": [],
            "context": "",
        }

        def stream_answer():
            # Retrieval data arrives before the generated answer tokens.
            for update in st.session_state.rag_chain.stream(question):
                if "source_pages" in update:
                    stream_state["source_pages"] = update["source_pages"]

                if "context" in update:
                    stream_state["context"] = update["context"]

                if "answer" in update:
                    yield update["answer"]

        try:
            with st.chat_message("assistant"):
                answer = st.write_stream(stream_answer())

                if not isinstance(answer, str) or not answer.strip():
                    answer = (
                        "I couldn't generate an answer. "
                        "Please try asking again."
                    )
                    st.markdown(answer)

                with st.expander("View retrieved sources"):
                    if stream_state["source_pages"]:
                        st.write(
                            "**Source pages:** "
                            + ", ".join(
                                str(page)
                                for page in stream_state["source_pages"]
                            )
                        )

                    if stream_state["context"]:
                        st.markdown("**Retrieved text**")
                        st.text(stream_state["context"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "source_pages": stream_state["source_pages"],
                    "context": stream_state["context"],
                }
            )

        except Exception as error:
            # Do not leave a failed exchange looking like a completed one.
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
                and st.session_state.messages[-1]["content"] == question
            ):
                st.session_state.messages.pop()

            st.error(
                "I couldn't complete the answer. "
                f"Check the terminal for details: {error}"
            )
