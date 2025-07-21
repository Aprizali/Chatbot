# config.py
from neo4j import GraphDatabase

# --- Konfigurasi Neo4j ---
URI = "neo4j+s://afd26999.databases.neo4j.io"  
AUTH = ("neo4j", "th-HaicSvvsZIY8oyK964w39JZ9BatwKrAtip7IkvlY") 

# --- Konfigurasi Embedding & Index ---
# Model yang digunakan di embedder.py
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

NODE_LABEL_FOR_INDEX = "Chunk"
INDEX_NAME = "vector_school_content_chunks_index" 
EMBEDDING_PROPERTY = "embedding"

# --- Konfigurasi Tokenizer & Chunking (digunakan oleh data_inserter.py) ---
MAX_TOKENS_PER_CHUNK = 512
TOKEN_OVERLAP = 50

# --- Inisialisasi Neo4j Driver ---
# Driver ini akan digunakan oleh data_inserter, vector_search, dan chatbot_ui
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        session.run("RETURN 1")
    print("INFO: Koneksi Neo4j berhasil diinisialisasi dari config.py.")
except Exception as e:
    print(f"ERROR: Gagal menginisialisasi koneksi Neo4j dari config.py: {e}")
    print("ERROR: Pastikan Neo4j server berjalan dan kredensial di config.py sudah benar.")
    driver = None

# --- Konfigurasi Groq (untuk chatbot_ui.py dan groq_client.py) ---
GROQ_API_KEY = "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" 
GROQ_MODEL = "llama-3.3-70b-versatile"
