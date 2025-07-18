import json
from config import driver, EMBEDDING_PROPERTY
from groq_embedder import GorqEmbedder

# Load data dari file JSON
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = GorqEmbedder()

with driver.session() as session:
    # Insert node Guru
    guru_embeddings = embedder.embed(data["guru"])
    for text, embedding in zip(data["guru"], guru_embeddings):
        session.run(
            f"""
            CREATE (n:Guru {{
                name: $text,
                {EMBEDDING_PROPERTY}: $embedding
            }})
            """,
            text=text,
            embedding=embedding
        )

    # Insert node Jawaban
    jawaban_embeddings = embedder.embed(data["jawaban"])
    for text, embedding in zip(data["jawaban"], jawaban_embeddings):
        session.run(
            f"""
            CREATE (n:Jawaban {{
                content: $text,
                {EMBEDDING_PROPERTY}: $embedding
            }})
            """,
            text=text,
            embedding=embedding
        )

print("✅ All nodes successfully inserted from dataset.json!")
