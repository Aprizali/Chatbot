# groq_client.py
import requests
import os
from config import GROQ_API_KEY, GROQ_MODEL

def generate_answer(context: str, question: str) -> str:
    """
    Menghasilkan jawaban menggunakan Groq API berdasarkan konteks dan pertanyaan.
    Prompt telah disempurnakan untuk menangani struktur data yang lebih kompleks.
    """
    effective_api_key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_") else os.environ.get("GROQ_API_KEY")

    if not effective_api_key:
        error_msg = "ERROR: GROQ_API_KEY tidak ditemukan di config.py atau environment variable. Tidak dapat menghubungi Groq."
        print(error_msg)
        return "Maaf, layanan AI tidak dapat dihubungi karena masalah konfigurasi API Key."
    
    # --- PROMPT ENGINEERING YANG DISEMPURNAKAN ---
    system_message_content = (
        "Anda adalah asisten AI dari SMK Medikacom Bandung. Anda sangat informatif, ramah, dan akurat. "
        "Tugas utama Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi yang ada di dalam 'KONTEKS YANG DITEMUKAN'.\n\n"
        "ATURAN UTAMA:\n"
        "1.  **JAWAB HANYA DARI KONTEKS**: Jangan pernah mengarang jawaban atau menggunakan pengetahuan di luar konteks yang diberikan.\n"
        "2.  **BAHASA**: Selalu gunakan Bahasa Indonesia yang baik dan jelas.\n"
        "3.  **JIKA TIDAK TAHU**: Jika informasi yang ditanyakan tidak ada di dalam konteks, jawab dengan jujur, contohnya: 'Maaf, saya tidak menemukan informasi mengenai [topik pertanyaan] dalam data yang saya miliki.'\n\n"
        "ATURAN PENYAJIAN JAWABAN SPESIFIK:\n"
        "-   **DAFTAR (Jurusan, Ekstrakurikuler, Staf)**: Jika pengguna meminta daftar (misalnya 'apa saja jurusan?', 'eskul apa saja?'), dan konteks menyediakan daftarnya, sajikan dalam format daftar bernomor atau poin (bullet points) agar mudah dibaca.\n"
        "-   **BIAYA PENDIDIKAN & SERAGAM**: Jika pertanyaan menyangkut biaya, selalu sebutkan untuk jurusan atau kelompok mana biaya tersebut berlaku. Rincikan komponen biaya (seperti DSP, SPP, item seragam) dan totalnya jika ada dalam konteks. Sebutkan juga skema cicilan jika informasinya tersedia.\n"
        "-   **KELAS INDUSTRI**: Jika ditanya tentang kelas industri (seperti Samsung atau Axioo), jelaskan secara lengkap mencakup jurusan terkait, manfaat yang didapat, biaya, dan kuota jika informasi tersebut ada di konteks.\n"
        "-   **PANDUAN PPDB**: Jika pertanyaan mengenai pendaftaran atau PPDB, jelaskan langkah-langkahnya secara berurutan sesuai alur yang ada di konteks.\n"
        "-   **FARMASI/KESEHATAN**: Jika pertanyaan mengandung kata kunci 'farmasi' atau 'kesehatan', fokuskan jawaban pada informasi yang relevan dengan jurusan kefarmasian yang ada di dalam konteks, seperti 'Layanan Penunjang Kefarmasian Klinis & Komunitas (FAR)', biaya, atau item seragam terkait.\n"
        "-   **FILTERING**: Jika pengguna menanyakan daftar dengan kriteria spesifik (contoh: 'siapa saja **guru** RPL?'), perhatikan baik-baik kata 'guru' dan saring dari daftar 'tenaga pendidik dan staf' untuk hanya menampilkan yang jabatannya adalah guru, bukan kepala sekolah atau staf.\n"
    )
    
    user_message_content = f"""
KONTEKS YANG DITEMUKAN:
\"\"\"
{context if context and context.strip() else "Tidak ada konteks yang relevan ditemukan dari basis data."}
\"\"\"

PERTANYAAN PENGGUNA:
{question}

JAWABAN (Berdasarkan aturan di atas, dalam Bahasa Indonesia, dan pastikan untuk menyaring daftar sesuai kriteria spesifik dalam pertanyaan jika diminta):
"""

    headers = {
        "Authorization": f"Bearer {effective_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ],
        "temperature": 0.1, # Dibuat sangat rendah untuk jawaban yang sangat faktual
        "max_tokens": 2048, # Cukup untuk jawaban yang detail
    }
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    print(f"INFO: Mengirim permintaan ke Groq API: {api_url} dengan model {GROQ_MODEL}")

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            answer = response_data["choices"][0].get("message", {}).get("content", "")
            print("INFO: Jawaban berhasil diterima dari Groq.")
            return answer.strip() if answer else "Maaf, saya tidak dapat menghasilkan jawaban saat ini."
        else:
            print(f"ERROR: Respons Groq tidak memiliki format yang diharapkan: {response_data}")
            return "Maaf, terjadi masalah saat memproses respons dari layanan AI."

    except requests.exceptions.HTTPError as http_err:
        print(f"ERROR: HTTP error saat menghubungi Groq API: {http_err}")
        print(f"ERROR: Response content: {http_err.response.text if http_err.response else 'No response content'}")
        return f"Maaf, terjadi kesalahan HTTP ({http_err.response.status_code if http_err.response else 'N/A'}) saat menghubungi layanan AI."
    except requests.exceptions.RequestException as req_err:
        print(f"ERROR: Kesalahan koneksi saat menghubungi Groq API: {req_err}")
        return "Maaf, terjadi kesalahan koneksi saat mencoba menghubungi layanan AI."
    except Exception as e:
        print(f"ERROR: Kesalahan tidak terduga saat memanggil Groq API: {e}")
        return "Maaf, terjadi kesalahan teknis yang tidak terduga."
