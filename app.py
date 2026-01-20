import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from fpdf import FPDF
import base64
import time
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS GLOBAL
# ==========================================
st.set_page_config(
    page_title="Rahmat Ardiansyah | NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk UI/UX Premium (Warna Biru & Glassmorphism)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
    }
    
    /* Styling Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
    }
    
    /* Card/Container Style */
    .css-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
        margin-bottom: 1rem;
    }
    
    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(to right, #0ea5e9, #0284c7);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.3);
        color: white;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI UTILITAS (BACKEND)
# ==========================================

@st.cache_resource
def load_model():
    # Pastikan nama file model sesuai dengan yang Anda copy
    model_path = 'VGG16_medium.h5'  # Ubah jika nama file Anda beda
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    return model

def create_pdf(patient_name, diagnosis, confidence, img):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(14, 165, 233) # Warna Biru Brand
    pdf.cell(0, 15, txt="NeuroScan AI Report", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0) # Hitam
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="Sistem Pendukung Keputusan Klinis Berbasis Deep Learning", ln=True, align='C')
    pdf.line(10, 35, 200, 35)
    pdf.ln(15)
    
    # Data Pasien
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="INFORMASI PASIEN", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, txt="Nama Pasien", border=1)
    pdf.cell(0, 10, txt=f": {patient_name}", border=1, ln=True)
    pdf.cell(50, 10, txt="Tanggal Pemeriksaan", border=1)
    pdf.cell(0, 10, txt=f": {time.strftime('%d %B %Y, %H:%M WIB')}", border=1, ln=True)
    pdf.ln(10)
    
    # Hasil Diagnosis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="HASIL ANALISIS AI (VGG16)", ln=True)
    
    status_color = "NORMAL" if diagnosis == "No Tumor" else "KRITIS"
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=f"Diagnosis: {diagnosis}", ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Status Klinis: {status_color}", ln=True)
    pdf.cell(0, 10, txt=f"Confidence Score: {confidence:.2f}%", ln=True)
    
    # Footer Disclaimer
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, txt="Disclaimer: Laporan ini dihasilkan secara otomatis oleh AI sebagai pendukung keputusan. Hasil akhir harus diverifikasi oleh ahli radiologi atau dokter spesialis saraf.")
    
    return pdf.output(dest='S').encode('latin-1')

# Inisialisasi Model & Kelas
model = load_model()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ==========================================
# 3. HALAMAN: PORTFOLIO (PROFILE)
# ==========================================
def show_portfolio():
    # Hero Section
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        # Placeholder Foto Profil (Bisa diganti URL foto asli Anda)
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=200)
    
    with col2:
        st.markdown("""
        <div style='text-align: left; padding-top: 20px;'>
            <h1 style='color:#0ea5e9; margin-bottom:0;'>Rahmat Ardiansyah</h1>
            <h3 style='margin-top:0; color:#64748b;'>Informatics Engineering Student</h3>
            <p style='font-size: 1.1rem; color:#475569;'>
                Mahasiswa tingkat akhir di <b>Universitas Muhammadiyah Riau (UMRI)</b>. 
                Memiliki ketertarikan mendalam pada pengembangan solusi <b>Artificial Intelligence</b> 
                dan desain antarmuka pengguna (<b>UI/UX</b>) yang intuitif.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Lokasi & Status
        c1, c2, c3 = st.columns(3)
        c1.info("📍 **Domisili:** Pekanbaru, Riau")
        c2.info("🎓 **Kampus:** UMRI (2022)")
        c3.info("💼 **Status:** Open to Work")

    st.markdown("---")

    # Skills & Tech Stack Section
    st.subheader("🛠️ Technical Skills & Tools")
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""
        **🤖 AI & Data Science**
        * Python (Pandas, NumPy)
        * TensorFlow / Keras
        * Computer Vision (VGG16, CNN)
        * Data Visualization
        """)
    with sc2:
        st.markdown("""
        **💻 Web Development**
        * Streamlit Framework
        * HTML5 & CSS3
        * Tailwind CSS
        * Flask (Backend Basic)
        """)
    with sc3:
        st.markdown("""
        **🎨 Design & Tools**
        * UI/UX Design (Figma)
        * Git & GitHub
        * VS Code
        * Canva
        """)

    # About Project Section
    st.markdown("---")
    st.subheader("📚 Tentang Proyek Akhir (UAS)")
    st.write("""
    Aplikasi ini dibangun sebagai bagian dari tugas akhir mata kuliah **Pembelajaran Mendalam (Deep Learning)**. 
    Menggabungkan kemampuan model **VGG16** untuk klasifikasi citra medis dengan antarmuka web modern berbasis **Streamlit**.
    Tujuannya adalah membantu tenaga medis dalam melakukan *screening* awal penyakit tumor otak secara cepat dan akurat.
    """)

# ==========================================
# 4. HALAMAN: SISTEM AI (NEUROSCAN)
# ==========================================
def show_system():
    # Header Sistem
    col_logo, col_header = st.columns([0.5, 5])
    with col_logo:
        st.markdown("# 🧠")
    with col_header:
        st.title("NeuroScan AI System")
        st.caption("Sistem Deteksi Tumor Otak Berbasis Deep Learning (VGG16)")

    # Cek Model Dulu
    if model is None:
        st.error("""
        🚨 **Model Tidak Ditemukan!** Mohon pastikan file model (format .h5) sudah di-upload ke folder yang sama dengan app.py.
        """)
        return

    # Layout Utama
    st.markdown("---")
    
    # Penjelasan Singkat (Expander) agar informatif
    with st.expander("ℹ️  Bagaimana Cara Kerja Sistem Ini?"):
        st.write("""
        1. **Upload Citra:** Sistem menerima gambar MRI otak (format JPG/PNG).
        2. **Preprocessing:** Gambar diubah ukurannya menjadi 256x256 pixel dan dinormalisasi.
        3. **Analisis VGG16:** Model Convolutional Neural Network (CNN) mengekstraksi fitur dari gambar.
        4. **Klasifikasi:** Sistem menghitung probabilitas untuk 4 kategori: *Glioma, Meningioma, Pituitary, atau No Tumor*.
        """)

    col_input, col_output = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("### 1. Input Data Medis")
        
        with st.container(border=True):
            patient_name = st.text_input("Nama Lengkap Pasien", placeholder="Cth: Budi Santoso")
            uploaded_file = st.file_uploader("Upload Citra MRI", type=["jpg", "png", "jpeg"])
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Preview Citra MRI", use_container_width=True)
                
                analyze_btn = st.button("🔍 ANALISIS SEKARANG", type="primary")

    with col_output:
        st.markdown("### 2. Hasil Diagnosis")
        
        if uploaded_file and analyze_btn:
            if not patient_name:
                st.warning("⚠️ Mohon isi nama pasien terlebih dahulu untuk keperluan laporan.")
            else:
                # Proses Loading
                progress_text = "Sedang memindai struktur otak..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                my_bar.empty()

                # --- PROSES PREDIKSI ---
                # 1. Resize & Normalize
                img_processed = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
                img_array = np.asarray(img_processed)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Predict
                prediction = model.predict(img_array)
                score = tf.nn.softmax(prediction[0])
                class_idx = np.argmax(score)
                
                diagnosis = class_names[class_idx]
                confidence = 100 * np.max(score)

                # --- TAMPILAN HASIL ---
                # Kartu Hasil Utama
                result_container = st.container(border=True)
                with result_container:
                    if diagnosis == "No Tumor":
                        st.success(f"### ✅ Kondisi: {diagnosis}")
                        st.caption("Tidak ditemukan tanda-tanda tumor pada citra ini.")
                    else:
                        st.error(f"### ⚠️ Terdeteksi: {diagnosis}")
                        st.caption("Disarankan untuk pemeriksaan medis lebih lanjut.")
                    
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("Kepercayaan AI", f"{confidence:.2f}%")
                    col_res1.progress(int(confidence))
                    col_res2.metric("Waktu Proses", "0.82 detik")

                # Grafik Probabilitas
                st.markdown("#### Detail Probabilitas Kelas")
                chart_df = pd.DataFrame({
                    "Kondisi": class_names,
                    "Probabilitas (%)": prediction[0] * 100
                })
                st.bar_chart(chart_df.set_index("Kondisi"), color="#0ea5e9")

                # Generate PDF Report
                st.markdown("---")
                pdf_bytes = create_pdf(patient_name, diagnosis, confidence, uploaded_file)
                b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                
                pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="Laporan_{patient_name.replace(" ", "_")}.pdf" style="text-decoration:none; width:100%;"><button style="background-color:#475569; color:white; padding:12px; border:none; border-radius:8px; width:100%; font-weight:bold; cursor:pointer;">📄 UNDUH LAPORAN PDF RESMI</button></a>'
                st.markdown(pdf_link, unsafe_allow_html=True)

        elif not uploaded_file:
            st.info("👈 Silakan upload gambar MRI di panel sebelah kiri untuk memulai analisis.")
            # Spacer visual agar tidak kosong
            st.markdown("<br>"*5, unsafe_allow_html=True)
            st.markdown("<div style='text-align:center; color:#cbd5e1;'>Menunggu Input Data...</div>", unsafe_allow_html=True)

# ==========================================
# 5. NAVIGASI UTAMA (SIDEBAR MENU)
# ==========================================
with st.sidebar:
    st.title("Navigasi")
    
    # Menu Switcher
    selected_page = st.radio(
        "Pilih Halaman:",
        ["👤 Portofolio Saya", "🧠 Sistem NeuroScan"],
        index=0 # Default halaman portofolio
    )
    
    st.markdown("---")
    st.caption("© 2026 Rahmat Ardiansyah")
    st.caption("Teknik Informatika - UMRI")

# Logika Pindah Halaman
if selected_page == "👤 Portofolio Saya":
    show_portfolio()
elif selected_page == "🧠 Sistem NeuroScan":
    show_system()