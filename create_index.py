# create_index.py
from config import driver, EMBEDDING_PROPERTY, NODE_LABEL_FOR_INDEX, INDEX_NAME
from embedder import GorqEmbedder

def create_vector_index_if_not_exists():
    temp_embedder = GorqEmbedder()
    dimension = temp_embedder.dimension
    del temp_embedder

    if dimension is None:
        print("❌ Tidak dapat menentukan dimensi embedding. Index tidak dibuat.")
        return

    print(f"ℹ️ Dimensi embedding yang terdeteksi dari model: {dimension}")

    with driver.session() as session:
        try:
            print(f"Membuat atau memastikan vector index '{INDEX_NAME}' ada untuk label '{NODE_LABEL_FOR_INDEX}' pada property '{EMBEDDING_PROPERTY}' dengan dimensi {dimension}...")
            
            session.run(f"""
                CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
                FOR (c:{NODE_LABEL_FOR_INDEX}) ON (c.{EMBEDDING_PROPERTY})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
            print(f"✅ Vector index '{INDEX_NAME}' berhasil dibuat atau sudah ada.")
        except Exception as e:
            print(f"❌ Gagal membuat vector index: {e}")
            if "Unsupported administration command" in str(e) or "Unknown command" in str(e):
                 print("ℹ️ Pastikan Anda menggunakan Neo4j versi 5.11 atau lebih baru untuk `CREATE VECTOR INDEX IF NOT EXISTS`.")
                 print("ℹ️ Atau, untuk versi lebih lama, hapus `IF NOT EXISTS` dan tangani error jika index sudah ada, atau buat index secara manual via Cypher Shell/Neo4j Browser.")
            elif "already exists" in str(e).lower():
                 print(f"ℹ️ Vector index '{INDEX_NAME}' sudah ada.")
        finally:
            driver.close()
            print("Koneksi ke Neo4j ditutup oleh create_index.")

if __name__ == "__main__":
    create_vector_index_if_not_exists()