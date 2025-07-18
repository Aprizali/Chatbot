# test_neo4j_conn.py
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
# PASTIKAN PASSWORD DI SINI SAMA PERSIS DENGAN YANG ADA DI CONFIG.PY DAN BENAR-BENAR PASSWORD NEO4J ANDA
AUTH = ("neo4j", "12345678") # Ganti "12345678" dengan password Anda yang sebenarnya

print(f"Mencoba menghubungkan ke Neo4j di {URI} dengan user '{AUTH[0]}'...")

try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        result = session.run("RETURN 1 AS test_result")
        record = result.single()
        print(f"Koneksi BERHASIL! Hasil query: {record['test_result']}")
    driver.close()
    print("Koneksi ditutup.")
except Exception as e:
    print(f"Koneksi GAGAL: {e}")
    if "authentication" in str(e).lower():
        print("KESALAHAN OTENTIKASI: Periksa kembali USERNAME dan terutama PASSWORD Anda.")
    elif "refused" in str(e).lower():
        print("KONEKSI DITOLAK: Pastikan server Neo4j berjalan dan dapat diakses di URI tersebut.")