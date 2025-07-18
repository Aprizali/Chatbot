# data_inserter.py
import json
from typing import List, Dict, Any, Optional
from neo4j import Driver, Transaction
from embedder import GorqEmbedder
from config import (
    NODE_LABEL_FOR_INDEX, 
    EMBEDDING_PROPERTY,
    MAX_TOKENS_PER_CHUNK,
    TOKEN_OVERLAP,
)
from transformers import PreTrainedTokenizerBase 

class Neo4jDataInserter:
    def __init__(self, driver: Driver, embedder: GorqEmbedder):
        self.driver = driver
        self.embedder = embedder
        if self.embedder and hasattr(self.embedder, 'get_tokenizer'):
            self.tokenizer: Optional[PreTrainedTokenizerBase] = self.embedder.get_tokenizer()
        else:
            self.tokenizer = None
            print("WARN: Embedder tidak memiliki metode get_tokenizer() atau embedder tidak terinisialisasi.")
        
        self.max_tokens = MAX_TOKENS_PER_CHUNK
        self.overlap = TOKEN_OVERLAP
        self.chunk_node_label = NODE_LABEL_FOR_INDEX

    def _tokenize_text(self, text: str) -> List[int]:
        if not self.tokenizer:
            print("ERROR: Tokenizer tidak tersedia. Menggunakan split by space sebagai fallback (kurang akurat).")
            return list(range(len(text.split()))) 
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if not self.tokenizer:
            print("ERROR: Tokenizer tidak tersedia. Tidak dapat decode token.")
            return " ".join(map(str, token_ids))
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _create_text_chunks(self, original_text: str, descriptive_prefix: str) -> List[str]:
        if not original_text or not original_text.strip():
            return []
        if not self.tokenizer:
            print(f"WARN: Tokenizer tidak tersedia. Tidak dapat melakukan chunking berbasis token untuk '{descriptive_prefix}'. Mengembalikan teks asli sebagai satu chunk.")
            return [f"{descriptive_prefix.strip()}: {original_text.strip()}"]

        full_text_to_process = f"{descriptive_prefix.strip()}: {original_text.strip()}"
        token_ids = self._tokenize_text(full_text_to_process)

        if not token_ids:
            return []

        chunks_as_strings: List[str] = []
        current_pos = 0
        while current_pos < len(token_ids):
            end_pos = min(current_pos + self.max_tokens, len(token_ids))
            chunk_token_ids = token_ids[current_pos:end_pos]
            chunk_text = self._decode_tokens(chunk_token_ids)
            chunks_as_strings.append(chunk_text)

            if end_pos == len(token_ids):
                break
            current_pos += (self.max_tokens - self.overlap)
            if current_pos >= end_pos and end_pos < len(token_ids): 
                current_pos = end_pos
        return chunks_as_strings

    def _insert_and_link_chunks(
        self,
        tx: Transaction,
        conceptual_parent_node_id: int,
        new_chunks_text: List[str],
        original_category_for_chunk: str,
        relationship_from_conceptual_to_first_chunk: str
    ) -> None:
        print(f"    INFO: Mencoba menghapus chunk lama untuk kategori '{original_category_for_chunk}' dari node ID {conceptual_parent_node_id} via rel :{relationship_from_conceptual_to_first_chunk}...")
        
        delete_query = f"""
        MATCH (p)-[:{relationship_from_conceptual_to_first_chunk}]->(old_first_chunk:{self.chunk_node_label})
        WHERE id(p) = $conceptual_parent_node_id
        OPTIONAL MATCH (old_first_chunk)-[:NEXT_CHUNK*0..]->(chunk_in_chain:{self.chunk_node_label})
        WITH collect(DISTINCT chunk_in_chain) AS all_chunks_to_delete_list
        UNWIND all_chunks_to_delete_list AS chunk_node_to_delete
        DETACH DELETE chunk_node_to_delete
        """

        delete_summary = tx.run(delete_query, conceptual_parent_node_id=conceptual_parent_node_id).consume()
        print(f"    ↪ INFO: {delete_summary.counters.nodes_deleted} node chunk lama dihapus untuk kategori '{original_category_for_chunk}'.")

        if not new_chunks_text:
            print(f"    INFO: Tidak ada chunk baru untuk disisipkan untuk kategori '{original_category_for_chunk}'.")
            return

        previous_chunk_id: Optional[int] = None
        for i, chunk_text_content in enumerate(new_chunks_text):
            embedding_vector = self.embedder.embed([chunk_text_content], for_query=False)[0]
            
            if not embedding_vector:
                print(f"ERROR: Gagal membuat embedding untuk chunk: '{chunk_text_content[:50]}...'. Chunk ini dilewati.")
                continue

            res = tx.run(
                f"""
                CREATE (c:{self.chunk_node_label} {{
                    text: $text,
                    {EMBEDDING_PROPERTY}: $embedding,
                    original_category: $original_category,
                    chunk_sequence: $chunk_sequence
                }})
                RETURN id(c) AS chunkId
                """,
                text=chunk_text_content,
                embedding=embedding_vector,
                original_category=original_category_for_chunk,
                chunk_sequence=i + 1,
            )
            current_chunk_id = res.single()["chunkId"]

            if i == 0: 
                tx.run(
                    f"""
                    MATCH (p) WHERE id(p) = $conceptualParentId
                    MATCH (c:{self.chunk_node_label}) WHERE id(c) = $chunkId
                    MERGE (p)-[:{relationship_from_conceptual_to_first_chunk}]->(c)
                    """,
                    conceptualParentId=conceptual_parent_node_id,
                    chunkId=current_chunk_id,
                )
            if previous_chunk_id is not None:
                tx.run(
                    f"""
                    MATCH (prev_c:{self.chunk_node_label}) WHERE id(prev_c) = $prevChunkId
                    MATCH (curr_c:{self.chunk_node_label}) WHERE id(curr_c) = $currChunkId
                    MERGE (prev_c)-[:NEXT_CHUNK]->(curr_c)
                    """,
                    prevChunkId=previous_chunk_id,
                    currChunkId=current_chunk_id,
                )
            previous_chunk_id = current_chunk_id
        print(f"    ↪ INFO: {len(new_chunks_text)} chunk baru disisipkan untuk '{original_category_for_chunk}' dan dihubungkan via :{relationship_from_conceptual_to_first_chunk} dari node konseptual ID {conceptual_parent_node_id}")

    def _ensure_conceptual_node(self, tx: Transaction, label: str, name_property: str, unique_name_value: str, other_props: Optional[Dict] = None) -> int:
        if other_props is None:
            other_props = {}
        
        all_props = {name_property: unique_name_value, **other_props}
        props_on_match = {k: v for k, v in other_props.items()}
        query = f"""
        MERGE (n:{label} {{{name_property}: $unique_name_value}})
        ON CREATE SET n = $all_props
        ON MATCH SET n += $props_on_match 
        RETURN id(n) as nodeId 
        """
        result = tx.run(query, unique_name_value=unique_name_value, all_props=all_props, props_on_match=props_on_match)
        node_id = result.single()["nodeId"]
        print(f"  INFO: Node konseptual dipastikan :{label} '{unique_name_value}' (ID: {node_id}).")
        return node_id

    def _ensure_relationship(self, tx: Transaction, from_node_id: int, to_node_id: int, rel_type: str):
        tx.run(
            f"""
            MATCH (a) WHERE id(a) = $from_id 
            MATCH (b) WHERE id(b) = $to_id 
            MERGE (a)-[:{rel_type}]->(b)
            """,
            from_id=from_node_id,
            to_id=to_node_id
        )

    def insert_data_from_json(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error saat memuat file JSON: {e}")
            return
        
        if not self.embedder or not self.tokenizer:
            print("❌ Error: Embedder atau tokenizer tidak terinisialisasi dengan benar. Proses insersi dibatalkan.")
            return

        with self.driver.session() as session:
            # 1. Node Sekolah Utama
            profil_sekolah_info = data.get("profil_sekolah", {})
            if not profil_sekolah_info.get("nama"):
                print("❌ Error: 'nama' sekolah tidak ditemukan dalam 'profil_sekolah'.")
                return
            
            sekolah_properties_to_set = {}
            for key, value in profil_sekolah_info.items():
                if key == "lokasi_lengkap" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        sekolah_properties_to_set[f"lokasi_{sub_key}"] = sub_value
                elif value is not None:
                    sekolah_properties_to_set[key] = value

            sekolah_node_id = session.execute_write(
                self._ensure_conceptual_node,
                "Sekolah", "nama", profil_sekolah_info["nama"],
                sekolah_properties_to_set
            )
            print(f"🏫 SEKOLAH UTAMA: '{profil_sekolah_info['nama']}' (ID: {sekolah_node_id})")

            # --- Mulai Proses Ingest per Kategori ---
            
            # 2. Deskripsi Profil Sekolah
            deskripsi_sekolah_parts = [f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in sekolah_properties_to_set.items() if key != 'nama']
            deskripsi_sekolah_text_full = f"Nama Sekolah: {profil_sekolah_info['nama']}. " + ". ".join(deskripsi_sekolah_parts)
            if deskripsi_sekolah_text_full:
                sekolah_desc_chunks = self._create_text_chunks(deskripsi_sekolah_text_full, "Deskripsi Profil Sekolah")
                session.execute_write(self._insert_and_link_chunks, sekolah_node_id, sekolah_desc_chunks, "Sekolah_DeskripsiProfil", "HAS_DESCRIPTION_CHUNK")

            # 3. Node Sejarah
            sejarah_list = data.get("sejarah", [])
            if sejarah_list:
                sejarah_konseptual_id = session.execute_write(self._ensure_conceptual_node, "Sejarah", "nama_kategori", "Sejarah Umum Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, sejarah_konseptual_id, "HAS_HISTORY")
                sejarah_text_full = " ".join(sejarah_list)
                sejarah_chunks = self._create_text_chunks(sejarah_text_full, "Sejarah Sekolah")
                session.execute_write(self._insert_and_link_chunks, sejarah_konseptual_id, sejarah_chunks, "Sejarah_Detail", "HAS_CONTENT_CHUNK")

            # 4. Node Visi & Misi
            vm_data = data.get("visi_dan_misi", {})
            if vm_data:
                visimisi_konseptual_id = session.execute_write(self._ensure_conceptual_node, "VisiMisi", "nama_kategori", "Visi dan Misi Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, visimisi_konseptual_id, "HAS_VISI_MISI")
                if vm_data.get("visi"):
                    visi_chunks = self._create_text_chunks(vm_data["visi"], "Visi Sekolah")
                    session.execute_write(self._insert_and_link_chunks, visimisi_konseptual_id, visi_chunks, "Visi_Detail", "HAS_VISI_CHUNK")
                if vm_data.get("misi"):
                    misi_text_full = " ".join([f"Poin Misi {i+1}: {m}" for i, m in enumerate(vm_data["misi"])])
                    misi_chunks = self._create_text_chunks(misi_text_full, "Misi Sekolah")
                    session.execute_write(self._insert_and_link_chunks, visimisi_konseptual_id, misi_chunks, "Misi_Detail", "HAS_MISI_CHUNK")

            # 5. Node Pengetahuan Umum
            pengetahuan_data = data.get("pengetahuan_umum", {})
            if pengetahuan_data:
                pu_konseptual_id = session.execute_write(self._ensure_conceptual_node, "PengetahuanUmum", "nama_kategori", "Pengetahuan Umum Jurusan")
                session.execute_write(self._ensure_relationship, sekolah_node_id, pu_konseptual_id, "HAS_GENERAL_KNOWLEDGE")
                
                pu_text_parts = []
                for jurusan, details in pengetahuan_data.items():
                    deskripsi = details.get('deskripsi', '')
                    karier = ', '.join(details.get('karier', []))
                    kuliah = ', '.join(details.get('kuliah_lanjutan', []))
                    fun_fact = details.get('fun_fact', '')
                    jurusan_text = f"Untuk jurusan {jurusan}, deskripsinya adalah: {deskripsi}. Prospek karier meliputi: {karier}. Pilihan kuliah lanjutan antara lain: {kuliah}. Fakta menarik: {fun_fact}."
                    pu_text_parts.append(jurusan_text)
                
                pu_text_full = " ".join(pu_text_parts)
                pu_chunks = self._create_text_chunks(pu_text_full, "Pengetahuan Umum tentang Jurusan")
                session.execute_write(self._insert_and_link_chunks, pu_konseptual_id, pu_chunks, "PengetahuanUmum_Detail", "HAS_CONTENT_CHUNK")

            # 6. Node Biaya Pendidikan
            biaya_data = data.get("biaya_pendidikan", {})
            if biaya_data:
                biaya_konseptual_id = session.execute_write(self._ensure_conceptual_node, "BiayaPendidikan", "nama_kategori", "Biaya Pendidikan Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, biaya_konseptual_id, "HAS_TUITION_INFO")
                
                biaya_text_parts = []
                if biaya_data.get("tahun_ajaran"): biaya_text_parts.append(f"Informasi biaya pendidikan untuk tahun ajaran {biaya_data['tahun_ajaran']}.")
                if biaya_data.get("catatan_umum"): biaya_text_parts.append(biaya_data['catatan_umum'])
                
                for kelompok in biaya_data.get("rincian_per_kelompok", []):
                    kelompok_parts = [f"Untuk kelompok keahlian {kelompok.get('kelompok_keahlian', 'N/A')}, total biaya tahun pertama adalah Rp {kelompok.get('total_biaya_tahun_pertama', 0):,}. Biaya ini sudah termasuk: {', '.join(kelompok.get('termasuk', []))}."]
                    skema_cicilan_parts = [f"pembayaran tahap {cicilan.get('tahap', '')} sebesar Rp {cicilan.get('jumlah', 0):,} dengan batas pembayaran {cicilan.get('batas_pembayaran', 'N/A')}" for cicilan in kelompok.get("skema_cicilan", [])]
                    if skema_cicilan_parts:
                        kelompok_parts.append(f"Skema pembayaran dapat dicicil: {', dan '.join(skema_cicilan_parts)}.")
                    biaya_text_parts.append(" ".join(kelompok_parts))
                
                pembayaran_non_tunai = biaya_data.get("metode_pembayaran_non_tunai", {})
                if pembayaran_non_tunai:
                    metode_parts = ["Metode pembayaran non-tunai yang tersedia adalah:"]
                    for bank_info in pembayaran_non_tunai.get("transfer_bank", []):
                        metode_parts.append(f"Transfer Bank {bank_info.get('bank')} ke nomor rekening {bank_info.get('nomor_rekening')} atas nama {bank_info.get('atas_nama')}.")
                    if pembayaran_non_tunai.get("qris"):
                        metode_parts.append(f"Pembayaran melalui QRIS juga tersedia, QR code dapat dilihat pada link {pembayaran_non_tunai.get('qris')}.")
                    biaya_text_parts.append(" ".join(metode_parts))

                biaya_text_full = " ".join(biaya_text_parts)
                biaya_chunks = self._create_text_chunks(biaya_text_full, "Rincian Biaya Pendidikan")
                session.execute_write(self._insert_and_link_chunks, biaya_konseptual_id, biaya_chunks, "BiayaPendidikan_Detail", "HAS_CONTENT_CHUNK")

            # 7. Node Biaya Seragam
            seragam_data = data.get("biaya_seragam", {})
            if seragam_data:
                seragam_konseptual_id = session.execute_write(self._ensure_conceptual_node, "BiayaSeragam", "nama_kategori", "Biaya Seragam Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, seragam_konseptual_id, "HAS_UNIFORM_COST_INFO")
                
                seragam_text_parts = []
                if seragam_data.get("tahun_ajaran"): seragam_text_parts.append(f"Informasi biaya seragam untuk tahun ajaran {seragam_data['tahun_ajaran']}.")
                
                for gender, rincian_gender in [("Pria", seragam_data.get("pria", [])), ("Wanita Muslim", seragam_data.get("wanita_muslim", []))]:
                    for kelompok in rincian_gender:
                        rincian_item_parts = [f"{item.get('item', '')} seharga Rp {item.get('biaya', 0):,}" for item in kelompok.get('rincian', [])]
                        kelompok_text = f"Biaya seragam untuk {gender} kelompok jurusan {kelompok.get('kelompok_jurusan', 'N/A')} adalah Rp {kelompok.get('total', 0):,} dengan rincian: {', '.join(rincian_item_parts)}."
                        seragam_text_parts.append(kelompok_text)
                
                seragam_text_full = " ".join(seragam_text_parts)
                seragam_chunks = self._create_text_chunks(seragam_text_full, "Rincian Biaya Seragam")
                session.execute_write(self._insert_and_link_chunks, seragam_konseptual_id, seragam_chunks, "BiayaSeragam_Detail", "HAS_CONTENT_CHUNK")

            # 8. Node Kelas Industri
            kelas_industri_data = data.get("kelas_industri", {})
            if kelas_industri_data:
                ki_konseptual_id = session.execute_write(self._ensure_conceptual_node, "KelasIndustri", "nama_kategori", "Kelas Industri Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, ki_konseptual_id, "HAS_INDUSTRY_CLASS")
                
                ki_text_parts = []
                if kelas_industri_data.get("catatan"): ki_text_parts.append(f"Catatan umum kelas industri: {kelas_industri_data['catatan']}.")
                for program in kelas_industri_data.get("program", []):
                    program_parts = [f"Program kelas industri bernama '{program.get('nama_program', 'N/A')}' ditujukan untuk jurusan {', '.join(program.get('jurusan_terkait', []))}. Kuota yang tersedia adalah {program.get('kuota', 'N/A')}."]
                    if program.get("total_biaya_tahun_pertama"): program_parts.append(f"Total biaya tahun pertama adalah Rp {program.get('total_biaya_tahun_pertama'):,}.")
                    if program.get("biaya_termasuk"): program_parts.append(f"Biaya tersebut sudah termasuk: {', '.join(program.get('biaya_termasuk', []))}.")
                    if program.get("manfaat"): program_parts.append(f"Manfaat yang didapat antara lain: {'. '.join(program.get('manfaat', []))}.")
                    skema_cicilan_ki_parts = [f"pembayaran tahap {cicilan.get('tahap', '')} sebesar Rp {cicilan.get('jumlah', 0):,} dengan batas pembayaran {cicilan.get('batas_pembayaran', 'N/A')}" for cicilan in program.get("skema_cicilan", [])]
                    if skema_cicilan_ki_parts:
                        program_parts.append(f"Skema pembayaran dapat dicicil: {', dan '.join(skema_cicilan_ki_parts)}.")
                    ki_text_parts.append(" ".join(program_parts))
                ki_text_full = " ".join(ki_text_parts)
                ki_chunks = self._create_text_chunks(ki_text_full, "Program Kelas Industri")
                session.execute_write(self._insert_and_link_chunks, ki_konseptual_id, ki_chunks, "KelasIndustri_Detail", "HAS_CONTENT_CHUNK")

            # 9. Node Tenaga Pendidik
            tp_list = data.get("tenaga_pendidik_dan_staf", [])
            if tp_list:
                tp_konseptual_id = session.execute_write(self._ensure_conceptual_node, "TenagaPendidik", "nama_kategori", "Tenaga Pendidik dan Staf Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, tp_konseptual_id, "HAS_STAFF")
                tp_text_full = ". ".join(tp_list)
                tp_chunks = self._create_text_chunks(tp_text_full, "Daftar Tenaga Pendidik dan Staf")
                session.execute_write(self._insert_and_link_chunks, tp_konseptual_id, tp_chunks, "TenagaPendidik_Detail", "HAS_CONTENT_CHUNK")

            # 10. Node Ekstrakurikuler
            eskul_list = data.get("ekstrakurikuler", [])
            if eskul_list:
                eskul_konseptual_id = session.execute_write(self._ensure_conceptual_node, "Ekstrakurikuler", "nama_kategori", "Ekstrakurikuler Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, eskul_konseptual_id, "HAS_EXTRACURRICULARS")
                eskul_text_full = ", ".join(eskul_list)
                eskul_chunks = self._create_text_chunks(eskul_text_full, "Daftar Ekstrakurikuler")
                session.execute_write(self._insert_and_link_chunks, eskul_konseptual_id, eskul_chunks, "Ekstrakurikuler_Detail", "HAS_CONTENT_CHUNK")

            # 11. Node Informasi Tambahan
            it_data = data.get("informasi_tambahan", {})
            if it_data:
                it_konseptual_id = session.execute_write(self._ensure_conceptual_node, "InformasiTambahan", "nama_kategori", "Informasi Tambahan Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, it_konseptual_id, "HAS_ADDITIONAL_INFO")
                for key, value in it_data.items():
                    descriptive_prefix = key.replace("_", " ").capitalize()
                    content_text = ", ".join(value) if isinstance(value, list) else str(value)
                    if content_text:
                        rel_name_suffix = ''.join(filter(str.isalnum, descriptive_prefix.replace(" ", "_")))
                        relationship_to_chunk = f"HAS_{rel_name_suffix.upper()}_CHUNK"
                        item_chunks = self._create_text_chunks(content_text, descriptive_prefix)
                        session.execute_write(self._insert_and_link_chunks, it_konseptual_id, item_chunks, f"InformasiTambahan_{key.capitalize()}", relationship_to_chunk)

            # 12. Node Panduan PPDB
            ppdb_data = data.get("panduan_ppdb", {})
            if ppdb_data:
                ppdb_konseptual_id = session.execute_write(self._ensure_conceptual_node, "PanduanPPDB", "nama_kategori", "Panduan PPDB Sekolah")
                session.execute_write(self._ensure_relationship, sekolah_node_id, ppdb_konseptual_id, "HAS_PPDB_GUIDE")
                
                ppdb_text_parts = []
                for kondisi, alur in ppdb_data.items():
                    alur_parts = [f"Untuk kondisi '{kondisi}', deskripsinya adalah: {alur.get('deskripsi', '')}."]
                    for langkah in alur.get("langkah_langkah", []):
                        tugas_utama = langkah.get('tugas_utama', langkah.get('tugas', ''))
                        langkah_part = [f"Langkah ke-{langkah.get('langkah_ke')}: {tugas_utama}."]
                        if "opsi_pembayaran" in langkah:
                            opsi_parts = []
                            for opsi in langkah['opsi_pembayaran']:
                                opsi_parts.append(f"Opsi pembayaran {opsi.get('metode')}: {opsi.get('detail')} menggunakan media {opsi.get('media')}.")
                            langkah_part.append(" ".join(opsi_parts))
                        else:
                            langkah_part.append(f"Media yang digunakan adalah {langkah.get('media', 'N/A')}.")
                        alur_parts.append(" ".join(langkah_part))
                    ppdb_text_parts.append(" ".join(alur_parts))
                
                ppdb_text_full = " ".join(ppdb_text_parts)
                ppdb_chunks = self._create_text_chunks(ppdb_text_full, "Panduan Pendaftaran Peserta Didik Baru (PPDB)")
                session.execute_write(self._insert_and_link_chunks, ppdb_konseptual_id, ppdb_chunks, "PanduanPPDB_Detail", "HAS_CONTENT_CHUNK")

            print("✅ Proses insersi data selesai.")
