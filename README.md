# ML_DeepFakeDetector
# Aplikasi Detektor Deepfake (Gambar, Video, & Audio)

> Proyek ini adalah aplikasi web yang mampu mendeteksi apakah sebuah **gambar wajah**, **video wajah**, atau **rekaman audio** merupakan hasil manipulasi **Deepfake** atau **Asli**. Aplikasi ini menggunakan model *deep learning* yang telah dilatih—**MobileNetV2** untuk gambar/video dan **CNN Spectrogram** untuk audio—untuk menganalisis file yang diunggah dan memberikan probabilitas keasliannya.
---
**Catatan Deployment:** Aplikasi ini dirancang untuk dijalankan secara lokal. Upaya *deployment* ke layanan cloud (Vercel, Railway) mengalami kendala teknis karena ukuran total proyek yang melebihi batas kapasitas yang disediakan (misalnya batas 300MB di Vercel). Oleh karena itu, aplikasi beroperasi penuh dalam lingkungan lokal.

## 👥 Anggota Kelompok
* Marco Darian Thomas(221112216) => Train Model
* Hugo Edri Chandra(221111848) => FrontEnd & Backend
* Valentino Karada(221110851) => Dokumentasi
---
## 🛠️ Teknologi yang Digunakan
Proyek ini dibangun menggunakan tumpukan teknologi berikut:
* **Backend:**
    * **Python 3.9+**
    * **Flask:** Sebagai *micro-framework* web untuk melayani API dan frontend.
    * **TensorFlow / Keras:** Untuk memuat dan menjalankan model *deep learning*.
    * **OpenCV (cv2):** Untuk membaca, membongkar, dan memproses file video *frame-by-frame*.
    * **Librosa:** Untuk pemrosesan audio dan konversi ke spectrogram.
    * **Pillow (PIL):** Untuk memproses file gambar.
    * **Numpy & Scikit-learn:** Untuk manipulasi data numerik dan evaluasi model.
* **Model AI:**
    * **MobileNetV2:** Untuk deteksi manipulasi pada gambar/frame video.
    * **CNN Spectrogram Kustom:** Untuk deteksi manipulasi pada file audio.
* **Frontend:**
    * HTML5, CSS3, JavaScript (fetch API)
---
## 🚀 Petunjuk Penggunaan Aplikasi
Aplikasi ini dapat mendeteksi gambar, video, dan audio:
1.  Buka aplikasi di browser (secara lokal di `http://127.0.0.1:5000`).
2.  Klik tombol **"Pilih File (Gambar/Video/Audio)"**.
3.  Pilih file (`.jpg`, `.png` untuk gambar; `.mp4` untuk video; `.wav`, `.mp3` untuk audio) dari komputer Anda.
4.  Pratinjau konten akan muncul di layar.
5.  Klik tombol **"Deteksi!"** untuk memulai analisis.
6.  Harap tunggu:
    * **Jika gambar:** Hasil akan muncul dalam beberapa detik.
    * **Jika video:** Proses akan memakan waktu lebih lama karena backend menganalisis beberapa frame dari video tersebut.
    * **Jika audio:** Proses melibatkan konversi ke spectrogram sebelum analisis.
7.  Hasil akhir (**Palsu (Fake)** atau **Asli (Real)**) akan ditampilkan beserta tingkat keyakinannya.
---
## ⚙️ Instalasi & Menjalankan Proyek di Lokal
Ikuti langkah-langkah ini untuk menginstal dan menjalankan salinan proyek ini di komputer lokal Anda.

### Prasyarat
Pastikan perangkat Anda telah terinstal perangkat lunak berikut:
* Git
* Python 3.9 atau yang lebih baru
* `pip` (Manajer paket Python)

### 1. Instalasi
1.  **Clone Repositori**
    Buka terminal Anda dan clone repositori ini:
    ```bash
    git clone https://github.com/KaradaVal/ML_DeepFakeDetector.git
    ```
2.  **Masuk ke Direktori Proyek**
    ```bash
    cd ML_DeepFakeDetector
    ```

3.  **Buat Virtual Environment (Sangat Direkomendasikan)**
    Ini akan mengisolasi dependensi proyek Anda.
    ```bash
    python -m venv venv
    # Di Windows:
    .\venv\Scripts\activate
    # Di Mac/Linux:
    source venv/bin/activate
    ```

4.  **Instal Dependensi**
    Pastikan Anda memiliki file `requirements.txt` di folder proyek dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependensi utama: flask tensorflow numpy pillow opencv-python-headless librosa scikit-learn)*

5.  **File Model**
    Pastikan file model terlatih untuk gambar/video (`deepfake_detector_finetuned.h5`) dan untuk audio (jika ada) berada di folder utama proyek, di samping `app.py`.

### 2. Menjalankan Proyek

1.  **Jalankan Server Flask**
    Setelah semua dependensi terinstal dan `venv` aktif, jalankan perintah berikut di terminal:
    ```bash
    python app.py
    ```

2.  **Buka Aplikasi**
    Server akan berjalan dan Anda akan melihat output seperti ini:
    ```
     * Running on http://127.0.0.1:5000
    ```
    Buka alamat `http://127.0.0.1:5000` di browser Anda untuk menggunakan aplikasi.