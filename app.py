import streamlit as st
import os
from utils import (
    load_embedding_model, read_file, read_url, generate_embedding,
    query_llm, save_uploaded_file, create_index, search_index,
    set_api_key, get_available_models
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to load custom CSS
def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def initialize_session_state():
    default_values = {
        'documents': [],
        'embeddings': [],
        'chat_history': [],
        'conversation_history': [],
        'index': None,
        'processed_files': [],
        'processed_urls': [],
        'clear_url': False,
        'selected_model': "",
        'selected_embedding_model': "",
        'api_key': os.getenv("GROQ_API_KEY", ""),
        'query_input': ""
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    st.set_page_config(
        page_title="Chatku AI",
        page_icon="🤖",
        layout="wide"
    )
    initialize_session_state()
    local_css("style.css")

    st.title("Retrieval Augmented Generation by Chatku AI")

    # Sidebar
    with st.sidebar:
        st.title("Chatku AI 🤖")
        handle_sidebar()

    # Main area
    handle_main_area()

def handle_sidebar():
    api_key = st.text_input("Masukkan Groq API Key:", value=st.session_state.api_key, type="password")
    if api_key:
        set_api_key(api_key)
        st.session_state.api_key = api_key
    else:
        st.warning("Masukkan Groq API yang valid agar bisa menggunakan Chatku AI")

    available_models = get_available_models()
    selected_model = st.selectbox("Pilih LLMs Model:", available_models, index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0)
    st.session_state.selected_model = selected_model

    embedding_models = ["all-MiniLM-L6-v2"]
    selected_embedding_model = st.selectbox("Pilih Embedding Model:", embedding_models, index=0)
    st.session_state.selected_embedding_model = selected_embedding_model

    uploaded_file = st.file_uploader("Pilih file (PDF or Word)", type=['pdf', 'docx'])
    url = st.text_input("Atau Masukkan URL", value="" if st.session_state.clear_url else st.session_state.get('url', ''))
    st.session_state.clear_url = False

    if uploaded_file:
        if st.button("Proses File"):
            process_file(uploaded_file)

    if url:
        if st.button("Proses URL"):
            process_url(url)

    display_processed_items()

    if st.button("Hasilkan Embeddings"):
        generate_embeddings(selected_embedding_model)

    if st.button("Buat Pencarian Index"):
        create_search_index()

    if st.button("Hasilkan Ringkasan Dokumen"):
        generate_document_summary(selected_model)

    if st.button("Export Chat History"):
        export_chat_history()

    if st.button("Hapus Semua Data"):
        clear_all_data()

def handle_main_area():
    st.subheader("Chatku AI")
    query = st.text_input("Masukkan Pertanyaan Anda", key="query_input")

    if query:
        handle_query(query)

    display_chat_history()

def process_file(uploaded_file):
    file_path = save_uploaded_file(uploaded_file)
    text = read_file(file_path)
    st.session_state.documents.append(text)
    st.session_state.processed_files.append(uploaded_file.name)
    st.sidebar.success(f"File '{uploaded_file.name}' proses berhasil!")

def process_url(url):
    text = read_url(url)
    st.session_state.documents.append(text)
    st.session_state.processed_urls.append(url)
    st.sidebar.success(f"URL '{url}' proses berhasil!")

def display_processed_items():
    if st.session_state.processed_files:
        st.write("Processed files:")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    if st.session_state.processed_urls:
        st.write("Processed URLs:")
        for url in st.session_state.processed_urls:
            st.write(f"- {url}")

def generate_embeddings(selected_embedding_model):
    if st.session_state.documents:
        model = load_embedding_model(selected_embedding_model)
        st.session_state.embeddings = []
        progress_bar = st.progress(0)
        for i, doc in enumerate(st.session_state.documents):
            embedding = generate_embedding(doc, model)
            st.session_state.embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(st.session_state.documents))
        st.success(f"Menghasilkan Embeddings untuk {len(st.session_state.embeddings)} dokumen")
    else:
        st.warning("Tidak ada dokumen untuk diproses. Silahkan masukkan file atau URL!")

def create_search_index():
    if len(st.session_state.embeddings) > 0:
        st.session_state.index = create_index(st.session_state.embeddings)
        st.success("Buat Search Index berhasil!")
    else:
        st.warning("Silahkan hasilkan Embeddings terlebih dahulu!")

def generate_document_summary(selected_model):
    if len(st.session_state.documents) > 0:
        summary_prompt = "Summarize the following documents:\n\n" + "\n\n".join(st.session_state.documents)
        with st.spinner("Menghasilkan Ringkasan..."):
            summary = query_llm(summary_prompt, selected_model)
        st.session_state.chat_history.append(("assistant", "Ringkasan Dokumen: " + summary))
        st.success("Ringkasan dokumen dibuat dan ditambahkan ke riwayat obrolan")
    else:
        st.warning("Silahkan proses dokumen terlebih dahulu!")

def export_chat_history():
    if st.session_state.chat_history:
        chat_export = "\n".join([f"{'User' if role == 'user' else 'Assistant'}: {message}" for role, message in st.session_state.chat_history])
        st.download_button(
            label="Download Chat History",
            data=chat_export,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    else:
        st.warning("Tidak ada riwayat obrolan untuk diekspor")

def clear_all_data():
    keys_to_clear = ['documents', 'embeddings', 'chat_history', 'conversation_history', 'processed_files', 'processed_urls', 'index']
    
    for key in keys_to_clear:
        if key in st.session_state:
            if isinstance(st.session_state[key], list):
                st.session_state[key] = []
            elif isinstance(st.session_state[key], dict):
                st.session_state[key] = {}
            else:
                st.session_state[key] = None

    st.session_state.clear_url = True
    if 'query_input' in st.session_state:
        st.session_state.query_input = ""

    st.success("Hapus semua data berhasil")
    st.experimental_rerun()

def handle_query(query):
    if not st.session_state.api_key:
        st.error("Silahkan masukkan Groq API Key untuk mengakses fitur Chatku AI")
        return

    st.session_state.conversation_history.append({"role": "user", "content": query})

    if st.session_state.index is not None:
        model = load_embedding_model(st.session_state.selected_embedding_model)
        query_embedding = generate_embedding(query, model)
        relevant_doc_indices = search_index(st.session_state.index, query_embedding)
        
        context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
        prompt = f"Based on the following context:\n\n{context}\n\nAnswer the following question: {query}"
    else:
        prompt = query

    full_prompt = "Previous conversation:\n"
    for message in st.session_state.conversation_history[:-1]:
        full_prompt += f"{message['role'].capitalize()}: {message['content']}\n"
    full_prompt += f"\nNew question: {prompt}\n\nPlease provide a response that takes into account the previous conversation."

    with st.spinner("Generating response..."):
        response = query_llm(full_prompt, st.session_state.selected_model)
    
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", response))

def display_chat_history():
    st.subheader("Chat History")
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <img src="https://cdn-icons-png.flaticon.com/512/1077/1077012.png" class="avatar" style="width: 50px; height: 50px;">
                    <div class="message"><b>User</b> <br>{message}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar" style="width: 50px; height: 50px;">
                    <div class="message"><b>Assistant</b> <br>{message}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()