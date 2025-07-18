# chatbot_ui.py
import streamlit as st

st.set_page_config(page_title="Chatbot MedikaCom", layout="wide", initial_sidebar_state="collapsed")

from neo4j_vector_search import Neo4jVectorSearcher
from embedder import GorqEmbedder
from groq_client import generate_answer
from config import driver as neo4j_driver

# --- Inisialisasi komponen ---
@st.cache_resource
def get_embedder():
    print("INFO: Menginisialisasi Embedder untuk UI...")
    return GorqEmbedder()

@st.cache_resource
def get_searcher(_driver, _embedder):
    print("INFO: Menginisialisasi Neo4jVectorSearcher untuk UI...")
    if not _driver:
        st.error("Koneksi Neo4j gagal diinisialisasi. Periksa config.py dan status server Neo4j.")
        return None
    if not _embedder or not _embedder.model:
        st.error("Embedder gagal diinisialisasi. Periksa model embedding.")
        return None
    return Neo4jVectorSearcher(driver_instance=_driver, embedder_instance=_embedder)

app_embedder = get_embedder()
app_searcher = get_searcher(neo4j_driver, app_embedder)

# --- UI Streamlit ---
st.title("🤖 Chatbot Informasi SMK MedikaCom")
st.markdown(
    """
    Selamat datang! Saya adalah chatbot yang dapat membantu Anda menemukan informasi seputar SMK Medikacom Bandung. 
    Silakan ajukan pertanyaan Anda mengenai jurusan, biaya, sejarah, visi misi, atau informasi lainnya.
    """
)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait SMK Medikacom?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Tulis pertanyaanmu di sini...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not app_searcher:
        st.error("Searcher tidak tersedia. Tidak dapat memproses permintaan.")
        st.stop()
    if not app_embedder or not app_embedder.model:
        st.error("Embedder atau model embedding tidak tersedia. Tidak dapat memproses permintaan.")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = ""
        
        with st.spinner("🔍 Mencari informasi di database..."):
            # top_k untuk mengambil lebih banyak chunk potensial
            retrieved_items = app_searcher.search_similar_chunks(user_input, top_k=10) 

        if not retrieved_items:
            context_for_llm = "Tidak ada informasi yang relevan ditemukan dalam basis data untuk pertanyaan ini."
            st.info("Hmm, sepertinya saya tidak menemukan informasi yang sangat cocok di database.")
        else:
            context_parts = []
            print("\n--- CHUNKS DIAMBIL UNTUK KONTEKS LLM ---")
            for i, item in enumerate(retrieved_items):
                print(f"Chunk {i+1} (Skor: {item.get('score', 'N/A'):.4f}, Kategori: {item.get('original_category', 'N/A')}):")
                print(f"  Teks: {item.get('text_content', '')[:200]}...") 
                context_parts.append(item.get('text_content', ''))
            
            context_for_llm = "\n\n---\n\n".join(context_parts)
            print("--- KONTEKS LENGKAP UNTUK LLM ---")
            print(context_for_llm) # Cetak konteks lengkap ke terminal
            print("---------------------------------")

        with st.spinner("🧠 Menyiapkan jawaban dengan AI..."):
            ai_response = generate_answer(context_for_llm, user_input)
            full_response_text = ai_response

        message_placeholder.markdown(full_response_text)

    st.session_state.messages.append({"role": "assistant", "content": full_response_text})

st.sidebar.title("Tentang Chatbot Ini")
st.sidebar.info(
    "Chatbot ini menggunakan model embedding untuk memahami pertanyaan Anda, "
    "mencari informasi relevan dari database Neo4j, dan kemudian menggunakan "
    "Large Language Model (LLM) dari Groq untuk menghasilkan jawaban berdasarkan informasi tersebut. "
)
st.sidebar.markdown("---")
if st.sidebar.button("Mulai Ulang Percakapan"):
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait SMK Medikacom?"}]
    st.rerun()
