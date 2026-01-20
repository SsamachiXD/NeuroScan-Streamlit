import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from fpdf import FPDF
import base64
import time
import os
import tempfile

# ===============================
# 1. KONFIGURASI TEMA & CSS CLOAKING
# ===============================
st.set_page_config(
    page_title="NeuroScan AI | Rahmat Ardiansyah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk meniru https://ssamachixd.github.io/NeuroScan_Tumor-Otak-MRI/
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Hero Section Gradient */
    .hero-container {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(14, 165, 233, 0.2);
    }

    /* Dashboard Cards */
    .saas-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    /* Navigation Bar */
    [data-testid="stSidebar"] {
        background-color: white !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Buttons */
    div.stButton > button {
        background: #0ea5e9 !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background: #0284c7 !important;
        transform: translateY(-2px);
    }

    /* Tags */
    .tag {
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
    }
    .tag-blue { background: #e0f2fe; color: #0369a1; }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. LOGIKA AI (VGG16)
# ===============================
@st.cache_resource
def load_neuro_model():
    path = "VGG16_medium.h5"
    if not os.path.exists(path): return None
    return tf.keras.models.load_model(path, compile=False)

model = load_neuro_model()
classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def create_pdf_report(name, diag, conf, img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(14, 165, 233)
    pdf.cell(0, 15, "NeuroScan Medical Report", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Diagnosis: {diag}", ln=True)
    pdf.cell(0, 10, f"Confidence Score: {conf:.2f}%", ln=True)
    pdf.cell(0, 10, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# ===============================
# 3. PAGES (UI/UX Portfolio Style)
# ===============================

def landing_page():
    st.markdown("""
    <div class='hero-container'>
        <h1 style='color:white; font-size:3rem; font-weight:800;'>NeuroScan AI</h1>
        <p style='font-size:1.2rem; opacity:0.9;'>Advanced Deep Learning for Clinical Radiology Support</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='saas-card'><h4>🔍 Akurasi Tinggi</h4><p>Menggunakan arsitektur VGG16 yang dioptimasi untuk citra medis.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='saas-card'><h4>📄 Laporan Instan</h4><p>Hasil diagnosis dapat diunduh langsung dalam format PDF resmi.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='saas-card'><h4>⚡ Proses Cepat</h4><p>Analisis citra MRI dilakukan dalam hitungan detik.</p></div>", unsafe_allow_html=True)

def system_page():
    st.markdown("## 🧠 Diagnostic Dashboard")
    
    if model is None:
        st.error("Model VGG16_medium.h5 tidak ditemukan.")
        return

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.subheader("Data Pasien")
        patient_name = st.text_input("Nama Lengkap")
        file = st.file_uploader("Upload Citra MRI", type=["jpg", "png", "jpeg"])
        analyze_btn = st.button("Jalankan Diagnosis AI")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Citra Input (MRI Scan)", use_container_width=True)
            
            if analyze_btn:
                if not patient_name:
                    st.warning("Mohon masukkan nama pasien.")
                else:
                    with st.spinner('Menganalisis...'):
                        time.sleep(1)
                        # Preprocessing sesuai spesifikasi VGG16_medium (150x150)
                        processed = ImageOps.fit(img, (150, 150), Image.Resampling.LANCZOS)
                        arr = np.asarray(processed) / 255.0
                        arr = np.expand_dims(arr, axis=0)
                        
                        pred = model.predict(arr)
                        idx = np.argmax(tf.nn.softmax(pred[0]))
                        diag = classes[idx]
                        conf = 100 * np.max(tf.nn.softmax(pred[0]))

                        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
                        color = "#22c55e" if diag == "No Tumor" else "#ef4444"
                        st.markdown(f"### Result: <span style='color:{color}'>{diag}</span>", unsafe_allow_html=True)
                        st.metric("Confidence Score", f"{conf:.2f}%")
                        st.bar_chart(pd.DataFrame(pred[0]*100, index=classes, columns=["%"]))
                        
                        pdf = create_pdf_report(patient_name, diag, conf, file)
                        b64 = base64.b64encode(pdf).decode()
                        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Laporan_{patient_name}.pdf">📥 Download Hasil PDF</a>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:400px; border:2px dashed #cbd5e1; border-radius:20px; display:flex; align-items:center; justify-content:center; color:#94a3b8;'>Silakan unggah citra MRI untuk memulai</div>", unsafe_allow_html=True)

def profile_page():
    st.markdown("## 👨‍💻 Developer Profile")
    st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        if os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", use_container_width=True)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", use_container_width=True)
    with c2:
        # Informasi personal mahasiswa Rahmat Ardiansyah
        st.markdown("### Rahmat Ardiansyah")
        st.markdown("<span class='tag tag-blue'>NIM: 220405010</span>", unsafe_allow_html=True)
        st.markdown("""
        Mahasiswa **Teknik Informatika - Universitas Muhammadiyah Riau (UMRI)**. 
        Memiliki fokus keahlian pada implementasi Deep Learning dalam sektor kesehatan digital.
        """)
        st.markdown("---")
        st.markdown("#### Technology Stack")
        st.info("Python • TensorFlow • Streamlit • Computer Vision (VGG16)")
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 4. ROUTER
# ===============================
def main():
    with st.sidebar:
        st.markdown("<h2 style='color:#0ea5e9;'>Menu Utama</h2>", unsafe_allow_html=True)
        page = st.radio("Navigasi", ["🏠 Home", "🧠 Diagnosis System", "👤 Developer Profile"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("NeuroScan AI v1.2")
        st.caption("Developed for UAS Project")

    if page == "🏠 Home": landing_page()
    elif page == "🧠 Diagnosis System": system_page()
    else: profile_page()

if __name__ == "__main__":
    main()