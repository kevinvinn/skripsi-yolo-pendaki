import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2  # Tambahkan ini untuk memperbaiki warna
import time # Tambahan library untuk menghitung waktu

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Logistik Pendaki", layout="wide")

# --- LOAD MODEL (Dengan Cache agar cepat) ---
@st.cache_resource 
def load_model():
    # Pastikan file model kamu namanya memang 'best.pt'
    return YOLO("best.pt")

model = load_model()

# --- JUDUL APLIKASI ---
st.title("🏔️ Sistem Pencatatan Logistik Pendaki Otomatis")
st.write("Gunakan aplikasi ini untuk mendeteksi dan mencatat barang bawaan logistik pendaki.")

# --- SIDEBAR (PENGATURAN MODEL) ---
st.sidebar.header("Pengaturan Model")
# Parameter Confidence Slider (sangat berguna saat sidang)
conf_threshold = st.sidebar.slider("Ambang Batas (Confidence Score)", 0.0, 1.0, 0.49)

# --- PILIHAN SUMBER DATA ---
source = st.radio("Pilih Sumber Gambar:", ("Unggah File (Foto)", "Ambil Foto (Kamera HP/Laptop)"))

image = None
# Logika pengambilan gambar
if source == "Unggah File (Foto)":
    uploaded_file = st.file_uploader("Upload foto logistik pendaki...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    img_file_buffer = st.camera_input("Arahkan kamera ke logistik pendaki")
    if img_file_buffer:
        image = Image.open(img_file_buffer)

# --- PROSES DETEKSI (Hanya jika gambar sudah ada) ---
if image is not None:
    # Tampilkan layout dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualisasi Deteksi")
        
        # --- MENGUKUR WAKTU DETEKSI ---
        start_time = time.time() # Catat waktu mulai
        
        # JALANKAN PREDIKSI: Masukkan objek gambar PIL ('image') LANGSUNG
        results = model.predict(source=image, conf=conf_threshold)
        
        end_time = time.time() # Catat waktu selesai
        
        # Hitung durasi inferensi dalam milidetik (ms)
        inference_time = (end_time - start_time) * 1000
        
        # Ambil Gambar Ber-Bounding Box (plot standard mengembalikan BGR)
        res_plotted = results[0].plot() 
        
        # --- PERBAIKAN WARNA ---
        # Ubah susunan warna dari BGR (OpenCV) kembali ke RGB (Browser/PIL asli)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Tampilkan di browser dengan warna asli dan info kecepatan di caption
        st.image(res_rgb, caption=f"Hasil Deteksi", use_container_width=True)
        
        # Tampilkan info teknis tambahan di bawah gambar
        st.info(f"⚡ Waktu Pemrosesan (Inference Time): {inference_time:.2f} ms")

    with col2:
        st.subheader("📋 Rekapitulasi Barang")
        
        # Ambil daftar ID kelas yang terdeteksi
        det_classes = results[0].boxes.cls.cpu().numpy()
        # Ambil daftar nama kelas asli dari model
        class_names = model.names
        # Terjemahkan ID menjadi Nama
        found_items = [class_names[int(cls)] for cls in det_classes]
        
        if found_items:
            # Hitung jumlah per barang menggunakan Pandas
            df = pd.DataFrame(found_items, columns=['Nama Barang'])
            summary_df = df['Nama Barang'].value_counts().reset_index()
            summary_df.columns = ['Nama Barang', 'Jumlah']
            
            # Tampilkan Tabel
            st.table(summary_df)
            
            # Fitur Download CSV (Nilai Tambah Skripsi)
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil (CSV)",
                data=csv,
                file_name='rekap_logistik.csv',
                mime='text/csv',
            )
            
            st.success(f"Terdeteksi total {len(found_items)} objek.")
        else:
            st.warning("Tidak ada objek yang terdeteksi. Coba turunkan Confidence Threshold di sidebar.")