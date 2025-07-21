# main.py
from config import driver  
from embedder import GorqEmbedder
from data_inserter import Neo4jDataInserter

def run_ingestion():
    print("🚀 Memulai proses ingestion data sekolah...")
    
    # Buat instance embedder
    my_embedder = GorqEmbedder()
    
    # Buat instance data inserter
    inserter = Neo4jDataInserter(driver, my_embedder)
    inserter.insert_data_from_json("dataset.json")
    driver.close()
    print("Koneksi ke Neo4j ditutup oleh main.")
    print("🎉 Proses ingestion selesai.")

if __name__ == "__main__":
    run_ingestion()