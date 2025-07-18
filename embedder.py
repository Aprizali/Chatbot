# embedder.py
from typing import List, cast
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class GorqEmbedder: # Nama kelas mungkin perlu disesuaikan jika tidak benar-benar menggunakan Groq untuk embedding
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Inisialisasi embedder dengan model SentenceTransformer.
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = cast(int, self.model.get_sentence_embedding_dimension())
            print(f"INFO: Embedder diinisialisasi dengan model: {model_name}, Dimensi: {self.dimension}")
        except Exception as e:
            print(f"ERROR: Gagal menginisialisasi SentenceTransformer model '{model_name}': {e}")
            print("ERROR: Pastikan model tersedia dan koneksi internet stabil jika model perlu diunduh.")
            self.model = None
            self.dimension = 0


    def embed(self, texts: List[str], for_query: bool = False) -> List[List[float]]:
        """
        Menghasilkan embeddings untuk daftar teks.
        
        Args:
            texts: Daftar string teks yang akan di-embed.
            for_query: Jika True, gunakan prefix "query: " (untuk pertanyaan pengguna).
                       Jika False, gunakan prefix "passage: " (untuk dokumen/chunk yang disimpan).
        """
        if not self.model:
            print("ERROR: Model embedding tidak terinisialisasi. Tidak dapat membuat embedding.")
            return [[] for _ in texts] # Kembalikan list of empty lists agar tidak error di pemanggil

        prefix = "query: " if for_query else "passage: "
        
        try:
            # print(f"DEBUG: Embedding dengan prefix '{prefix}': {texts[:1]}...") # Untuk debugging
            embeddings = self.model.encode(
                [f"{prefix}{text}" for text in texts],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat membuat embedding: {e}")
            return [[] for _ in texts]


    def get_tokenizer(self):
        """Mengembalikan tokenizer dari model SentenceTransformer."""
        if not self.model:
            print("ERROR: Model embedding tidak terinisialisasi. Tidak dapat mengambil tokenizer.")
            return None
        return self.model.tokenizer
