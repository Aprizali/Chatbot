# config.py
from neo4j import GraphDatabase

# --- Konfigurasi Neo4j ---
URI = "neo4j+s://afd26999.databases.neo4j.io"  # Ganti jika Neo4j Anda berjalan di host/port berbeda
AUTH = ("neo4j", "th-HaicSvvsZIY8oyK964w39JZ9BatwKrAtip7IkvlY") # GANTI DENGAN KREDENSIAL NEO4J ANDA

# --- Konfigurasi Embedding & Index ---
# Model yang digunakan di embedder.py
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# NODE_LABEL_FOR_INDEX adalah label untuk node yang di-embed dan akan diindeks.
# Dalam struktur baru kita, ini adalah "Chunk".
NODE_LABEL_FOR_INDEX = "Chunk"
# INDEX_NAME harus sama dengan yang dibuat oleh create_index.py
INDEX_NAME = "vector_school_content_chunks_index" 
EMBEDDING_PROPERTY = "embedding" # Nama properti di node :Chunk yang menyimpan vector

# --- Konfigurasi Tokenizer & Chunking (digunakan oleh data_inserter.py) ---
MAX_TOKENS_PER_CHUNK = 512
TOKEN_OVERLAP = 50

# --- Inisialisasi Neo4j Driver ---
# Driver ini akan digunakan oleh data_inserter, vector_search, dan chatbot_ui
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    # Uji koneksi sederhana
    with driver.session() as session:
        session.run("RETURN 1")
    print("INFO: Koneksi Neo4j berhasil diinisialisasi dari config.py.")
except Exception as e:
    print(f"ERROR: Gagal menginisialisasi koneksi Neo4j dari config.py: {e}")
    print("ERROR: Pastikan Neo4j server berjalan dan kredensial di config.py sudah benar.")
    driver = None # Set driver ke None jika gagal

# --- Konfigurasi Groq (untuk chatbot_ui.py dan groq_client.py) ---
# GANTI DENGAN API KEY GROQ ANDA YANG VALID
GROQ_API_KEY = "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" 
GROQ_MODEL = "llama-3.3-70b-versatile" # Model Groq yang ingin Anda gunakan
# GROQ_MODEL = "mixtral-8x7b-32768" # Alternatif model jika diperlukan
