from typing import List
from sentence_transformers import SentenceTransformer

class GorqEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        
        return self.model.encode(
            [f"passage: {text}" for text in texts],
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
