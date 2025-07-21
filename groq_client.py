# groq_client.py
import requests
import os
import time # Import library 'time'
from config import GROQ_API_KEY, GROQ_MODEL

def generate_answer(context: str, question: str) -> str:
    """
    Menghasilkan jawaban menggunakan Groq API berdasarkan konteks dan pertanyaan.
    Ditambahkan jeda untuk menghindari rate limiting.
    """
    # --- SOLUSI: TAMBAHKAN JEDA DI SINI ---
    # Memberi jeda 3 detik sebelum setiap panggilan API untuk mengurangi kemungkinan rate limit.
    # Anda bisa menonaktifkan baris ini jika tidak diperlukan lagi.
    print("INFO: Menunggu 3 detik sebelum mengirim ke API untuk menghindari rate limit...")
    time.sleep(3)
    
    effective_api_key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_") else os.environ.get("GROQ_API_KEY")

    if not effective_api_key:
        error_msg = "ERROR: GROQ_API_KEY tidak ditemukan di config.py atau environment variable. Tidak dapat menghubungi Groq."
        print(error_msg)
        return "Maaf, layanan AI tidak dapat dihubungi karena masalah konfigurasi API Key."
    
    system_message_content = (
        "Anda adalah asisten AI dari SMK Medikacom Bandung. Anda sangat informatif, ramah, dan akurat. "
        "Tugas utama Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi yang ada di dalam 'KONTEKS YANG DITEMUKAN'.\n\n"
        "ATURAN UTAMA:\n"
        "1.  **JAWAB HANYA DARI KONTEKS**: Jangan pernah mengarang jawaban atau menggunakan pengetahuan di luar konteks yang diberikan.\n"
        "2.  **BAHASA**: Selalu gunakan Bahasa Indonesia yang baik dan jelas.\n"
        "3.  **INFERENSI CERDAS**: Jika pengguna menanyakan sesuatu dengan istilah tertentu (misalnya 'biaya pendaftaran'), dan konteks berisi informasi yang sangat relevan dengan istilah yang sedikit berbeda (misalnya 'biaya pendidikan'), jawablah secara langsung dan percaya diri menggunakan informasi yang relevan tersebut. Jangan meminta maaf karena tidak menemukan istilah yang persis sama.\n"
        "4.  **JIKA TIDAK TAHU**: Jika informasi yang ditanyakan benar-benar tidak ada atau tidak relevan sama sekali di dalam konteks, barulah jawab dengan jujur, contohnya: 'Maaf, saya tidak menemukan informasi mengenai [topik pertanyaan] dalam data yang saya miliki.'\n\n"
        "ATURAN FORMAT JAWABAN:\n"
        "-   **GUNAKAN MARKDOWN UNTUK DAFTAR**: Jika jawaban Anda berisi daftar item (seperti rincian biaya, jurusan, ekstrakurikuler), SELALU gunakan format daftar Markdown (bullet points) dengan tanda '-' atau '*' di awal setiap item.\n"
        "-   **RINCIAN BIAYA**: Saat menjawab pertanyaan tentang biaya, rincikan untuk setiap jurusan yang relevan. Tampilkan total biayanya, lalu berikan detail komponennya dalam sub-daftar. Selalu sebutkan skema cicilan jika informasinya tersedia.\n"
        "-   **FILTERING GURU BERDASARKAN JURUSAN**: Jika pengguna menanyakan daftar guru untuk jurusan tertentu (contoh: 'siapa saja guru akuntansi?'), Anda HARUS menyaring daftar 'tenaga pendidik dan staf' untuk hanya menampilkan nama yang deskripsinya mengandung kata 'Guru' DAN juga nama atau singkatan jurusan yang ditanyakan.\n"
    )
    
    user_message_content = f"""
KONTEKS YANG DITEMUKAN:
\"\"\"
{context if context and context.strip() else "Tidak ada konteks yang relevan ditemukan dari basis data."}
\"\"\"

PERTANYAAN PENGGUNA:
{question}

JAWABAN (Berdasarkan semua aturan di atas, dalam Bahasa Indonesia, dan gunakan format daftar Markdown untuk rincian):
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
        "temperature": 0.1,
        "max_tokens": 2048,
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