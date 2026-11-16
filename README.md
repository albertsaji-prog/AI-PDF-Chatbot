
# AI PDF Question Answering Chatbot

## Project Description
An AI-powered chatbot that allows users to upload PDF documents and ask questions based on their contents.

## Technologies Used
- Python
- Streamlit
- Ollama
- Llama 3.2
- Nomic Embeddings
- NumPy
- PyPDF

## Features
- Upload PDF documents
- Extract text from PDF
- Split text into chunks
- Generate embeddings
- Retrieve relevant information using RAG
- Generate answers using LLM

## How to Run

1. Clone this repository.
2. Install Python.
3. Install Ollama.
4. Download the required models:

   ollama pull llama3.2:3b

   ollama pull nomic-embed-text

5. Install dependencies:

   pip install -r requirements.txt

6. Run the application:

   streamlit run app.py

## Developed Using
Python, Streamlit, Ollama and Retrieval-Augmented Generation (RAG).
