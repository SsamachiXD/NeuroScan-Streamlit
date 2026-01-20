import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
import os
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Rahmat.Dev | NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM UNTUK TAMPILAN ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #0ea5e9;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #334155;}
    .card {background-color: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px;}
    .profile-name {font-size: 2rem; font-weight: 800; color: #0f172a;}
    .tech-badge {
        display: inline-block; padding: 5px 10px; margin: 3px; 
        background-color: #e0f2fe; color: #0284c7; 
        border-radius: 15px; font-size: 0.85rem; font-weight: 600;
    }
    .footer {text-align: center; margin-top: 50px; color: #64748b; font-size: 0.8rem;}
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model_path = 'VGG16_medium.h5'
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# --- FUNGSI BANTUAN ---
def preprocess_image(image):
    # Sesuaikan ukuran dengan input VGG16 (biasanya 224x224 atau sesuai training Anda)
    size = (224, 224) 
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisasi jika diperlukan (misal 1/255)
    img_array = img_array.astype(np.float32) / 255.0
    
    # Expand dims (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def create_pdf(patient_name, diagnosis, confidence, probabilities):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="NeuroScan AI - Medical Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Nama Pasien: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Tanggal: {st.session_state.get('date', 'Hari ini')}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Hasil Diagnosis:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Kondisi: {diagnosis}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Level: {confidence:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Rincian Probabilitas:", ln=True)
    for cls, prob in probabilities.items():
        pdf.cell(200, 10, txt=f"- {cls}: {prob:.2f}%", ln=True)
        
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(200, 10, txt="*Laporan ini dihasilkan oleh AI dan bukan pengganti diagnosis dokter profesional.", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR (NAVBAR) ---
with st.sidebar:
    st.markdown("## R | Rahmat.Dev")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigasi",
        ["Portfolio", "NeuroScan AI"],
        index=0
    )
    
    st.markdown("---")
    st.caption("© 2026 Rahmat Ardiansyah")

# --- HALAMAN 1: PORTFOLIO ---
if selected_page == "Portfolio":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Tampilkan Foto Profil jika ada
        if os.path.exists("rahmat2.jpg"):
            st.image("rahmat2.jpg", caption="", use_container_width=True)
        elif os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", caption="", use_container_width=True)
        else:
            st.markdown("📷 *Foto Profil*")
            
    with col2:
        st.markdown('<p class="sub-header">Halo, Saya</p>', unsafe_allow_html=True)
        st.markdown('<p class="profile-name">Rahmat Ardiansyah</p>', unsafe_allow_html=True)
        
        st.write("""
        Mahasiswa Teknik Informatika di Universitas Muhammadiyah Riau dengan ketertarikan mendalam 
        pada pengembangan **UI/UX Design** dan **Artificial Intelligence**.
        """)
        
        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.markdown("**📍 Domisili**")
            st.write("Pekanbaru, Riau")
        with c_info2:
            st.markdown("**🎓 Universitas**")
            st.write("UMRI (2022)")
            
        st.markdown("### 🛠 Tech Stack & Tools")
        tech_stacks = ["C++", "JavaScript", "HTML5", "Figma", "Canva", "Tailwind", "TensorFlow", "Python", "Streamlit"]
        
        # Render badges
        badges_html = "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_stacks])
        st.markdown(badges_html, unsafe_allow_html=True)

# --- HALAMAN 2: NEUROSCAN AI ---
elif selected_page == "NeuroScan AI":
    st.markdown('<p class="main-header">🧠 NeuroScan AI Dashboard</p>', unsafe_allow_html=True)
    
    st.info("""
    **Sistem Pendukung Keputusan Klinis (CDSS)** Berbasis Deep Learning (VGG16) untuk mendeteksi dan mengklasifikasikan 4 kondisi otak:
    * Glioma Tumor
    * Meningioma Tumor
    * Pituitary Tumor
    * No Tumor (Normal)
    """)
    
    # Layout 2 Kolom: Kiri (Input), Kanan (Output)
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        st.markdown("### 📂 Data & Upload")
        with st.container(border=True):
            patient_name = st.text_input("Nama Pasien", placeholder="Masukkan nama...")
            uploaded_file = st.file_uploader("Upload MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])
            
            consent = st.checkbox("Saya menyetujui pemrosesan data medis ini.")
            
            analyze_btn = st.button("JALANKAN DIAGNOSIS", type="primary", use_container_width=True)
            
            if st.button("RESET", use_container_width=True):
                st.rerun()

    with right_col:
        st.markdown("### 📊 Hasil Analisis")
        
        if analyze_btn:
            if not model:
                st.error("❌ Model AI tidak ditemukan. Pastikan file 'VGG16_medium.h5' ada.")
            elif not uploaded_file:
                st.warning("⚠️ Silakan upload gambar MRI terlebih dahulu.")
            elif not patient_name:
                st.warning("⚠️ Silakan isi nama pasien.")
            elif not consent:
                st.warning("⚠️ Harap setujui pemrosesan data medis.")
            else:
                try:
                    # Tampilkan progress
                    with st.spinner('Sedang menganalisis citra MRI...'):
                        image = Image.open(uploaded_file).convert('RGB')
                        processed_img = preprocess_image(image)
                        
                        # Prediksi
                        predictions = model.predict(processed_img)
                        idx = np.argmax(predictions[0])
                        confidence = float(predictions[0][idx] * 100)
                        diagnosis = CLASS_NAMES[idx]
                        
                        # Dictionary Probabilitas
                        probs = {name: float(prob * 100) for name, prob in zip(CLASS_NAMES, predictions[0])}

                    # Tampilkan Hasil
                    st.success("✅ Diagnosis Selesai")
                    
                    # Kartu Hasil Utama
                    st.markdown(f"""
                    <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border: 1px solid #bbf7d0; text-align: center;">
                        <h3 style="color: #166534; margin:0;">{diagnosis}</h3>
                        <p style="color: #15803d; font-weight: bold; margin:0;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan Gambar
                    st.image(image, caption=f"MRI Scan - {patient_name}", width=200)
                    
                    # Grafik Probabilitas
                    st.markdown("**Probabilitas Kelas:**")
                    st.bar_chart(probs, color="#0ea5e9")
                    
                    # Tombol Download PDF
                    pdf_bytes = create_pdf(patient_name, diagnosis, confidence, probs)
                    st.download_button(
                        label="📄 Download Laporan PDF",
                        data=pdf_bytes,
                        file_name=f"Laporan_Medis_{patient_name.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                    )
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                    
        else:
            # Tampilan Default (Belum ada aksi)
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #94a3b8; border: 2px dashed #cbd5e1; border-radius: 10px;">
                <h4>Sistem Siap</h4>
                <p>Silakan lengkapi data dan upload gambar MRI di panel kiri untuk memulai diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div class="footer">
    © 2026 Rahmat Ardiansyah. Teknik Informatika UMRI.<br>
    Developed with Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)