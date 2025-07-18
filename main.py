# main.py
from config import driver  # Hanya driver yang diimpor dari config
from embedder import GorqEmbedder
from data_inserter import Neo4jDataInserter

def run_ingestion():
    print("🚀 Memulai proses ingestion data sekolah...")
    
    # Buat instance embedder
    my_embedder = GorqEmbedder()
    
    # Buat instance data inserter
    inserter = Neo4jDataInserter(driver, my_embedder)
    
    # Jalankan proses insersi dari file JSON
    # Pastikan dataset.json ada di direktori yang sama atau berikan path yang benar
    inserter.insert_data_from_json("dataset.json")
    
    # Tutup driver Neo4j setelah selesai
    driver.close()
    print("Koneksi ke Neo4j ditutup oleh main.")
    print("🎉 Proses ingestion selesai.")

if __name__ == "__main__":
    # Penting: Jalankan create_index.py terlebih dahulu jika index belum ada!
    # Dianjurkan menjalankan create_index.py sebagai langkah terpisah.
    # print(" ಮೊದಲು 'create_index.py' ಅನ್ನು ರನ್ ಮಾಡುವುದನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ! ") # "Pastikan untuk menjalankan 'create_index.py' terlebih dahulu!" dalam bahasa Kannada
    # print("Pastikan untuk menjalankan 'create_index.py' terlebih dahulu jika index belum ada atau konfigurasi berubah!")

    run_ingestion()