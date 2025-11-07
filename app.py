import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import cv2  # Library baru untuk memproses video
import uuid # Untuk membuat nama file sementara yang unik

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
try:
    model = load_model(MODEL_PATH)
    print(f"--- Model berhasil dimuat dari {MODEL_PATH} ---")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan file '{MODEL_PATH}' ada.")
    print(f"Error detail: {e}")
    exit()

LABEL_MAP = {0: 'Palsu (Fake)', 1: 'Asli (Real)'}
IMG_SIZE = (224, 224) # Ukuran input model Anda
TEMP_FOLDER = 'temp_uploads' # Folder untuk menyimpan video sementara
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# -------------------------------------------------------------------
# Fungsi Preprocessing
# -------------------------------------------------------------------

def preprocess_image_from_pil(img_pil):
    """ Memproses gambar dari object PIL.Image """
    try:
        # 1. Pastikan 3 Channel (RGB)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # 2. Resize ke (224, 224)
        img_pil = img_pil.resize(IMG_SIZE)
        
        # 3. Konversi ke array NumPy
        img_array = np.array(img_pil)
        
        # 4. Tambah dimensi batch (menjadi 1, 224, 224, 3)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # 5. Gunakan pre-processing bawaan MobileNetV2
        return preprocess_input(img_array_expanded)
    
    except Exception as e:
        print(f"Error saat preprocessing PIL: {e}")
        return None

def preprocess_frame_from_cv2_for_batching(frame):
    """ 
    Memproses frame CV2 (dari BGR) dan me-resize-nya.
    Siap untuk ditumpuk (batching).
    """
    try:
        # 1. Konversi BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Resize ke ukuran input model (lebih cepat pakai INTER_AREA)
        resized_frame = cv2.resize(rgb_frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
        return resized_frame
    except Exception as e:
        print(f"Error saat preprocessing frame CV2: {e}")
        return None

# -------------------------------------------------------------------
# Rute API untuk Prediksi GAMBAR
# -------------------------------------------------------------------
@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file gambar terkirim'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'})

    try:
        # Buka gambar dari stream
        img_pil = Image.open(file.stream)
        
        # Proses gambar
        processed_image = preprocess_image_from_pil(img_pil)
        
        if processed_image is None:
            return jsonify({'success': False, 'error': 'Gagal memproses gambar'})

        # Lakukan prediksi
        prediction = model.predict(processed_image, verbose=0)
        score = float(prediction[0][0])
        
        if score > 0.5:
            class_id = 1 # Asli
            confidence = score * 100
        else:
            class_id = 0 # Palsu
            confidence = (1 - score) * 100
            
        predicted_label = LABEL_MAP[class_id]

        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': f'{confidence:.2f}%',
            'raw_score': score,
            'type': 'image'
        })

    except Exception as e:
        print(f"Error di endpoint /predict_image: {e}")
        return jsonify({'success': False, 'error': f'Terjadi kesalahan server: {e}'})

# -------------------------------------------------------------------
# Rute API untuk Prediksi VIDEO (Logika Cepat)
# -------------------------------------------------------------------
@app.route('/predict_video', methods=['POST'])
def predict_video_route():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file video terkirim'})
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'})

    # Buat nama file sementara yang aman
    temp_filename = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.mp4")
    
    try:
        # 1. Simpan video ke file sementara
        file.save(temp_filename)
        
        # 2. Buka video menggunakan OpenCV
        cap = cv2.VideoCapture(temp_filename)
        
        # --- PERBAIKAN 1: Pengambilan Frame yang Lebih Baik ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Tentukan jumlah frame yang ingin dianalisis (misal: 30)
        NUM_FRAMES_TO_ANALYZE = 30
        
        frame_indices = []
        if total_frames > 0:
            if total_frames < NUM_FRAMES_TO_ANALYZE:
                frame_indices = np.arange(0, total_frames)
            else:
                frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_ANALYZE, dtype=int)
        
        frames_to_predict = []
        print(f"Memproses video: {temp_filename}. Total frame: {total_frames}. Menganalisis {len(frame_indices)} frame.")
        
        # --- PERBAIKAN 2: Loop untuk Mengumpulkan Frame ---
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            processed_frame = preprocess_frame_from_cv2_for_batching(frame)
            if processed_frame is not None:
                frames_to_predict.append(processed_frame)
        
        cap.release()
        
        # --- PERBAIKAN 3: Prediksi Batch (JAUH LEBIH CEPAT) ---
        if not frames_to_predict:
            return jsonify({'success': False, 'error': 'Gagal membaca frame dari video.'})
        
        # Tumpuk semua frame menjadi satu batch NumPy
        batch_frames = np.array(frames_to_predict)
        
        # Normalisasi seluruh batch sekaligus (penting!)
        normalized_batch = preprocess_input(batch_frames)
        
        # Lakukan prediksi HANYA SATU KALI pada seluruh batch
        predictions_list = model.predict(normalized_batch, verbose=0)
        
        print(f"Selesai memproses. Total frame dianalisis: {len(predictions_list)}")

        # 7. Analisis hasil
        scores = predictions_list.flatten()
        avg_score = np.mean(scores)
        
        if avg_score > 0.5:
            class_id = 1 # Asli
            confidence = avg_score * 100
        else:
            class_id = 0 # Palsu
            confidence = (1 - avg_score) * 100
        
        predicted_label = LABEL_MAP[class_id]

        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': f'{confidence:.2f}%',
            'raw_score': avg_score,
            'type': 'video',
            'frames_analyzed': len(predictions_list)
        })

    except Exception as e:
        print(f"Error di endpoint /predict_video: {e}")
        return jsonify({'success': False, 'error': f'Terjadi kesalahan server: {e}'})
    
    finally:
        # 8. Hapus file video sementara, apapun yang terjadi
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# -------------------------------------------------------------------
# Rute untuk Halaman Utama (Frontend)
# -------------------------------------------------------------------
@app.route('/')
def home():
    # Sajikan file index.html dari folder 'templates'
    return render_template('index.html')

# -------------------------------------------------------------------
# Jalankan Aplikasi
# -------------------------------------------------------------------
if __name__ == '__main__':
    # threaded=False PENTING untuk kompatibilitas Keras/Flask
    app.run(debug=True, port=5000, threaded=False)