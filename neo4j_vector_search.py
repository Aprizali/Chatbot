# neo4j_vector_search.py
from typing import List, Dict, Any, Optional
from neo4j import Driver, Session
from embedder import GorqEmbedder
from config import driver as default_driver, INDEX_NAME as DEFAULT_INDEX_NAME

class Neo4jVectorSearcher:
    SEQUENTIAL_CATEGORIES = [
        "TenagaPendidik_Detail", "Misi_Detail", "Sejarah_Detail",
        "Visi_Detail", "Sekolah_DeskripsiProfil",
        "InformasiTambahan_Akreditasi", "InformasiTambahan_BiayaPendaftaran",
        "InformasiTambahan_Fasilitas", "InformasiTambahan_JamOperasionalSekolah",
        "InformasiTambahan_KegiatanEkstrakurikuler", "InformasiTambahan_KontakPenting",
        "InformasiTambahan_PendaftaranSiswaBaru", "InformasiTambahan_PrestasiSekolah",
        "InformasiTambahan_ProgramBeasiswa", "InformasiTambahan_TataTertibSekolah",
        "InformasiTambahan_TransportasiUmum", "InformasiTambahan_SeragamSekolah"
        
    ]

    def __init__(self, driver_instance: Driver, embedder_instance: GorqEmbedder, index_name: str = DEFAULT_INDEX_NAME):
        if not driver_instance:
            raise ValueError("Neo4j driver instance tidak boleh None.")
        if not embedder_instance:
            raise ValueError("Embedder instance tidak boleh None.")
        self.driver = driver_instance
        self.embedder = embedder_instance
        self.vector_index_name = index_name
        print(f"INFO: Neo4jVectorSearcher diinisialisasi untuk index: '{self.vector_index_name}'")

    def _find_sequence_head_id(self, session: Session, any_chunk_id_in_sequence: int, original_category_of_sequence: str) -> Optional[int]:
        """
        Mencari ID dari chunk pertama (head) dalam suatu urutan.
        Query ini mengasumsikan 'head_chunk' memiliki chunk_sequence = 1 dan terhubung ke 'conceptual_parent'
        yang juga merupakan induk dari 'any_chunk_id_in_sequence'.
        """
        query = """
            MATCH (any_chunk:Chunk)
            WHERE id(any_chunk) = $any_chunk_id_in_sequence AND any_chunk.original_category = $original_category_of_sequence
            MATCH (conceptual_parent)-[r_to_first_chunk]->(head_chunk:Chunk {chunk_sequence: 1, original_category: $original_category_of_sequence})
            MATCH (head_chunk)-[:NEXT_CHUNK*0..]->(any_chunk)
            RETURN id(head_chunk) AS head_id
            LIMIT 1
        """
        try:
            result = session.run(query, any_chunk_id_in_sequence=any_chunk_id_in_sequence, original_category_of_sequence=original_category_of_sequence).single()
            return result["head_id"] if result and result["head_id"] else None
        except Exception as e:
            print(f"ERROR saat mencari sequence head untuk chunk_id {any_chunk_id_in_sequence}, category {original_category_of_sequence}: {e}")
            return None

    def _get_full_sequence_from_head(self, session: Session, head_chunk_id: int, target_original_category: str) -> List[str]:
        """Mengambil semua teks dari suatu urutan, dimulai dari head_chunk_id-nya."""
        cypher_query = """
            MATCH (head_chunk:Chunk)
            WHERE id(head_chunk) = $head_chunk_id AND head_chunk.original_category = $target_original_category
            MATCH path=(head_chunk)-[:NEXT_CHUNK*0..]->(sequence_chunk:Chunk)
            WHERE sequence_chunk.original_category = $target_original_category
            WITH sequence_chunk, length(path) AS depth
            ORDER BY depth
            RETURN sequence_chunk.text AS text_content
        """
        try:
            result = session.run(cypher_query, head_chunk_id=head_chunk_id, target_original_category=target_original_category)
            return [record["text_content"] for record in result]
        except Exception as e:
            print(f"ERROR saat mengambil sequence dari head_id {head_chunk_id}, category {target_original_category}: {e}")
            return []

    def _format_context_item(self, text_content: str, category: str, score: float, source_type: str,
                               vss_hit_chunk_id: int, sequence_head_id: Optional[int] = None) -> Dict[str, Any]:
        item = {
            "text_content": text_content,
            "original_category": category,
            "score": score,
            "source_type": source_type,
            "retrieved_via_chunk_id": vss_hit_chunk_id
        }
        if sequence_head_id:
            item["sequence_head_id"] = sequence_head_id
        return item

    def search_similar_chunks(self, user_question: str, top_k: int = 10) -> List[Dict[str, Any]]: # top_k bisa disesuaikan
        if not user_question:
            print("WARN: Pertanyaan pengguna kosong, mengembalikan hasil kosong.")
            return []

        query_embedding = self.embedder.embed([user_question], for_query=True)[0]
        if not query_embedding:
            print("ERROR: Gagal membuat embedding. Pencarian dibatalkan.")
            return []

        cypher_query_initial_search = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding_vector)
        YIELD node, score
        RETURN node.text AS text_content, 
               node.original_category AS original_category, 
               score, 
               id(node) AS nodeId,
               COALESCE(node.chunk_sequence, -1) AS chunk_sequence // -1 jika tidak ada chunk_sequence
        ORDER BY score DESC
        """
        
        print(f"DEBUG: Menjalankan vector search: '{user_question[:50]}...', top_k: {top_k}")
        initial_retrieved_chunks: List[Dict[str, Any]] = []
        final_llm_contexts: List[Dict[str, Any]] = []
        processed_ids = set() 

        try:
            with self.driver.session(database="neo4j") as session: 
                result = session.run(
                    cypher_query_initial_search,
                    index_name=self.vector_index_name, top_k=top_k, embedding_vector=query_embedding
                )
                initial_retrieved_chunks = [dict(record) for record in result]

                for vss_hit_chunk_data in initial_retrieved_chunks:
                    category = vss_hit_chunk_data["original_category"]
                    vss_hit_chunk_id = vss_hit_chunk_data["nodeId"]
                    score = vss_hit_chunk_data["score"]
                    
                    unique_item_id_to_check = None 
                    is_sequential = category in self.SEQUENTIAL_CATEGORIES

                    if is_sequential:
                        head_id = self._find_sequence_head_id(session, vss_hit_chunk_id, category)
                        if head_id:
                            unique_item_id_to_check = head_id
                        else:
                            print(f"WARN: Tidak menemukan head untuk chunk sekuensial ID {vss_hit_chunk_id}, kategori {category}. Dilewati.")
                            continue 
                    else:
                        unique_item_id_to_check = vss_hit_chunk_id 
                    if unique_item_id_to_check in processed_ids:
                        
                        print(f"DEBUG: Item dengan ID unik {unique_item_id_to_check} sudah diproses. Melewati VSS hit chunk {vss_hit_chunk_id}.")
                        continue

                    if is_sequential and unique_item_id_to_check: 
                        full_sequence_texts = self._get_full_sequence_from_head(session, unique_item_id_to_check, category)
                        if full_sequence_texts:
                            combined_text = " ".join(full_sequence_texts)
                            final_llm_contexts.append(self._format_context_item(
                                combined_text, category, score, "expanded_sequence", vss_hit_chunk_id, unique_item_id_to_check
                            ))
                            processed_ids.add(unique_item_id_to_check) 
                        else:
                            print(f"WARN: Gagal mengambil sequence dari head_id {unique_item_id_to_check} (dari VSS hit {vss_hit_chunk_id}).")
                    elif not is_sequential and unique_item_id_to_check: 
                        final_llm_contexts.append(self._format_context_item(
                            vss_hit_chunk_data["text_content"], category, score, "original_chunk_non_sequential", vss_hit_chunk_id, None
                        ))
                        processed_ids.add(unique_item_id_to_check) 
        
        except Exception as e:
            print(f"ERROR: Saat vector search atau pemrosesan hasil: {e}")
            return final_llm_contexts if 'final_llm_contexts' in locals() else []


        # Logging hasil akhir
        print(f"INFO: [RETRIEVAL LOG] Pertanyaan: '{user_question}'")
        log_msg = f"Awalnya {len(initial_retrieved_chunks)} chunk dari VSS. "
        log_msg += f"Setelah ekspansi & deduplikasi menjadi {len(final_llm_contexts)} blok konteks."
        print(f"INFO: Index '{self.vector_index_name}'. {log_msg}")

        for i, item in enumerate(final_llm_contexts, 1):
            text = item.get('text_content', '').replace('\n', ' ').replace('\r', '')[:500]
            print(f"  {i}. Skor: {item['score']:.4f} | Kategori: {item.get('original_category')} ({item.get('source_type')}) | Teks: {text}...")
        
        return final_llm_contexts

if __name__ == "__main__":
    from config import driver as main_test_driver
    
    if not main_test_driver:
        print("ERROR: Neo4j driver tidak terinisialisasi di config.py.")
    else:
        print("INFO: Menjalankan contoh Neo4jVectorSearcher...")
        try:
            my_embedder = GorqEmbedder() 
            if not hasattr(my_embedder, 'model') or not my_embedder.model: 
                 raise Exception("Embedder model gagal dimuat atau tidak memiliki atribut 'model'.")

            searcher = Neo4jVectorSearcher(main_test_driver, my_embedder)
            test_questions = [
                "sebutin semua guru yang ada di smk medikacom",
                "Apa visi sekolah medikacom?",
                "Bagaimana sejarah singkat sekolah medikacom?"
            ]
            for q_idx, test_question in enumerate(test_questions):
                print(f"\n--- Menguji Pertanyaan {q_idx+1}: '{test_question}' ---")
                similar_docs = searcher.search_similar_chunks(test_question, top_k=7) # top_k bisa disesuaikan

                if similar_docs:
                    print(f"\n[HASIL FINAL UNTUK '{test_question}'] ({len(similar_docs)} blok):")
                    for doc_idx, doc in enumerate(similar_docs):
                        print(f"  Dokumen {doc_idx+1}: Skor: {doc['score']:.4f}, Kategori: {doc.get('original_category')}, Source: {doc.get('source_type')}")
                        if "sequence_head_id" in doc: print(f"    Sequence Head ID: {doc['sequence_head_id']}")
                        text_preview = doc.get('text_content', '').replace('\n', ' ').replace('\r', '')[:200]
                        print(f"    Teks: {text_preview}...\n")
                else:
                    print(f"Tidak ada dokumen cocok untuk '{test_question}'.")
        
        except Exception as e:
            print(f"ERROR menjalankan contoh: {e}")
        finally:
            if main_test_driver:
                main_test_driver.close()
                print("\nINFO: Koneksi Neo4j (dari contoh) ditutup.")