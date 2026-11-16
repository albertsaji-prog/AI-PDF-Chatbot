
import streamlit as st
import ollama
import numpy as np
import hashlib
from pypdf import PdfReader

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions using AI and RAG.")

# -------------------------------
# MODEL CONFIGURATION
# -------------------------------

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# -------------------------------
# TEXT EXTRACTION
# -------------------------------

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -------------------------------
# TEXT CHUNKING
# -------------------------------

def split_text(text, chunk_size=180, overlap=40):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -------------------------------
# CREATE EMBEDDINGS
# -------------------------------

def create_embeddings(chunks):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=[
            "search_document: " + chunk
            for chunk in chunks
        ]
    )

    return np.array(response.embeddings)


# -------------------------------
# RAG RETRIEVAL
# -------------------------------

def retrieve_relevant_chunks(question, chunks, embeddings):
    response = ollama.embed(
        model=EMBED_MODEL,
        input="search_query: " + question
    )

    question_embedding = np.array(response.embeddings[0])

    # Calculate cosine similarity
    scores = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_embedding)
        + 1e-10
    )

    # Select top 3 relevant chunks
    top_indices = np.argsort(scores)[::-1][:3]

    results = []

    for index in top_indices:
        results.append(chunks[index])

    return results


# -------------------------------
# LLM ANSWER GENERATION
# -------------------------------

def generate_answer(question, context):

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the
    information provided in the context.

    If the answer is not available in the
    context, say:
    "The answer is not available in the PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# -------------------------------
# SESSION STATE
# -------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_id" not in st.session_state:
    st.session_state.file_id = None


# -------------------------------
# PDF UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:

    file_id = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.file_id != file_id:

        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.messages = []
        st.session_state.file_id = None

    if st.button("Process PDF"):

        try:
            with st.spinner("Reading and processing PDF..."):

                text = extract_text(uploaded_file)

                if not text.strip():
                    st.error(
                        "No readable text found. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                chunks = split_text(text)

                embeddings = create_embeddings(chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.file_id = file_id

                st.success("PDF processed successfully!")

                st.write("Total text chunks:", len(chunks))

        except Exception as e:
            st.error(f"Error processing PDF: {e}")


# -------------------------------
# CHAT INTERFACE
# -------------------------------

if st.session_state.embeddings is not None:

    st.divider()

    st.subheader("💬 Ask Questions")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("Finding information and generating answer..."):

                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings
                )

                context = "\n\n".join(relevant_chunks)

                answer = generate_answer(
                    question,
                    context
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

                with st.expander("View retrieved information"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"Source chunk {i + 1}:")
                        st.write(chunk)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:
            st.error(f"Error generating answer: {e}")

else:
    st.info("Please upload and process a PDF to start chatting.")

st.divider()

st.caption("Built using Python, Streamlit, Ollama, Llama 3.2 and RAG.")
