import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import streamlit as st

groq_client = None

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def set_api_key(api_key):
    global groq_client
    groq_client = Groq(api_key=api_key)

def get_available_models():
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

@st.cache_data
def read_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def generate_embedding(text, model):
    return model.encode(text)

def query_llm(prompt, model_name):
    if groq_client is None:
        return "Please enter a valid API key first."
    try:
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while querying the LLM: {str(e)}"

def save_uploaded_file(uploaded_file):
    file_path = os.path.join('data', uploaded_file.name)
    os.makedirs('data', exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def create_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_index(index, query_embedding, k=3):
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]