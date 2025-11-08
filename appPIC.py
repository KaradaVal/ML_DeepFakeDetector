import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io

# -------------------------------------------------------------------
# Inisialisasi Aplikasi Flask
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Konfigurasi & Pemuatan Model
# -------------------------------------------------------------------

# Tentukan path ke model Anda
MODEL_PATH = 'deepfake_detector_finetuned.h5'

# Muat model saat aplikasi dimulai
# (Ini bisa memakan waktu beberapa detik)
try:
    model = load_model(MODEL_PATH)
    print(f"--- Model berhasil dimuat dari {MODEL_PATH} ---")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan file '{MODEL_PATH}' ada.")
    print(f"Error detail: {e}")
    # Jika model gagal dimuat, kita tidak bisa melanjutkan
    exit()

# Tentukan mapping label (sesuai dengan folder training Anda)
# Sesuaikan ini jika 'training_fake' Anda bukan 0
LABEL_MAP = {0: 'Palsu (Fake)', 1: 'Asli (Real)'}
IMG_SIZE = (224, 224) # Ukuran input model Anda

# -------------------------------------------------------------------
# Fungsi Preprocessing (PENTING!)
# -------------------------------------------------------------------
def preprocess_image(image_file_stream):
    """
    Fungsi ini mengambil stream file gambar, memprosesnya 
    agar sesuai dengan input model MobileNetV2.
    """
    try:
        # 1. Buka file gambar
        img = Image.open(image_file_stream)
        
        # 2. Pastikan 3 Channel (RGB), buang transparansi (RGBA -> RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 3. Resize ke (224, 224)
        img = img.resize(IMG_SIZE)
        
        # 4. Konversi ke array NumPy
        img_array = np.array(img)
        
        # 5. Tambah dimensi batch (menjadi 1, 224, 224, 3)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # 6. Gunakan pre-processing bawaan MobileNetV2
        return preprocess_input(img_array_expanded)
        
    except Exception as e:
        print(f"Error saat preprocessing: {e}")
        return None

# -------------------------------------------------------------------
# Rute API untuk Prediksi
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Cek apakah ada file 'image' dalam request
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file gambar terkirim'})
    
    file = request.files['image']
    
    # 2. Cek apakah nama file kosong
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'})

    try:
        # 3. Proses gambar
        processed_image = preprocess_image(file.stream)
        
        if processed_image is None:
            return jsonify({'success': False, 'error': 'Gagal memproses gambar'})

        # 4. Lakukan prediksi
        prediction = model.predict(processed_image)
        
        # 5. Interpretasi hasil
        # Hasil 'prediction' adalah nilai sigmoid (cth: 0.98 atau 0.02)
        score = float(prediction[0][0])
        
        if score > 0.5:
            class_id = 1 # Asli
            confidence = score * 100
        else:
            class_id = 0 # Palsu
            confidence = (1 - score) * 100
            
        predicted_label = LABEL_MAP[class_id]

        # 6. Kembalikan hasil sebagai JSON
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': f'{confidence:.2f}%',
            'raw_score': score
        })

    except Exception as e:
        print(f"Error di endpoint /predict: {e}")
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan server: {e}'
        })

# -------------------------------------------------------------------
# Rute untuk Halaman Utama (Frontend)
# -------------------------------------------------------------------
@app.route('/')
def home():
    # Sajikan file index.html
    return render_template('index.html')

# -------------------------------------------------------------------
# Jalankan Aplikasi
# -------------------------------------------------------------------
if __name__ == '__main__':
    # threaded=False PENTING untuk kompatibilitas Keras/Flask
    app.run(debug=True, port=5000, threaded=False)