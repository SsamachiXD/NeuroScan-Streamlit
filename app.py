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

# ==========================================
# 1. KONFIGURASI HALAMAN & FIX CSS
# ==========================================
st.set_page_config(
    page_title="NeuroScan AI | Rahmat Ardiansyah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS INJECTION: MEMAKSA WARNA TEKS TETAP GELAP (ANTI-DARK MODE CONFLICT)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Text & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f8fafc !important;
        color: #1e293b !important; /* Warna teks utama gelap */
    }

    /* Memastikan teks markdown, judul, dan label berwarna gelap */
    .stMarkdown, p, span, label, h1, h2, h3, h4 {
        color: #1e293b !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #1e293b !important;
    }

    /* Card Container */
    .metric-card {
        background-color: white !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        color: #1e293b !important;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #0ea5e9 !important;
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.4);
    }
    div.stButton > button:hover {
        background-color: #0284c7 !important;
        transform: translateY(-2px);
    }

    /* Form Inputs */
    .stTextInput input {
        color: #1e293b !important;
        background-color: white !important;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIKA BACKEND
# ==========================================

@st.cache_resource
def load_model_ai():
    model_path = 'VGG16_medium.h5'
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_clinical_pdf(patient_name, diagnosis, confidence, img_obj):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Laporan
    pdf.set_fill_color(240, 249, 255)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_y(10)
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, txt="NeuroScan AI", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, txt="Laporan Analisis Radiologi Digital", ln=True, align='C')
    pdf.ln(20)
    
    # Data Pasien
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, txt="DATA PASIEN", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(50, 10, txt="Nama Pasien", border='B')
    pdf.cell(0, 10, txt=f": {patient_name}", border='B', ln=True)
    pdf.cell(50, 10, txt="Waktu Scan", border='B')
    pdf.cell(0, 10, txt=f": {time.strftime('%d-%m-%Y %H:%M WIB')}", border='B', ln=True)
    pdf.ln(10)
    
    # Preview Citra di PDF
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(img_obj).convert('RGB')
            img.save(tmp.name)
            pdf.image(tmp.name, x=75, w=60)
            pdf.ln(5)
    except:
        pdf.cell(0, 10, txt="[Gambar MRI]", ln=True, align='C')

    # Hasil
    pdf.set_font("Arial", 'B', 14)
    color = (34, 197, 94) if diagnosis == "No Tumor" else (239, 68, 68)
    pdf.set_text_color(*color)
    pdf.cell(0, 15, txt=f"DIAGNOSIS: {diagnosis.upper()}", ln=True, border=1, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)
    pdf.ln(5)
    pdf.cell(0, 8, txt=f"Tingkat Kepercayaan: {confidence:.2f}%", ln=True)
    
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 5, txt="Penafian: Hasil AI adalah pendukung keputusan, bukan diagnosis medis final.")
    
    return pdf.output(dest='S').encode('latin-1')

model = load_model_ai()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ==========================================
# 3. KOMPONEN UI
# ==========================================

def show_portfolio():
    st.markdown("## 👨‍💻 Developer Profile")
    col1, col2 = st.columns([1, 3], gap="medium")
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Rahmat Ardiansyah</h4>
            <p>Mahasiswa Teknik Informatika <b>Universitas Muhammadiyah Riau (UMRI)</b>.</p>
            <hr>
            <p>NIM: 220405010<br>Minat: Deep Learning & UI/UX Design</p>
        </div>
        """, unsafe_allow_html=True)

def show_system():
    st.markdown("<h1>NeuroScan AI System</h1>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Model tidak ditemukan.")
        return

    col_ctrl, col_res = st.columns([1, 1.5], gap="large")

    with col_ctrl:
        st.subheader("1. Input Data")
        with st.container(border=True):
            name = st.text_input("Nama Pasien", placeholder="Nama lengkap...")
            file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])
            btn = st.button("🔍 ANALISIS SEKARANG", type="primary", use_container_width=True)

    with col_res:
        st.subheader("2. Hasil Analisis")
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Citra Input", use_container_width=True)
            
            if btn:
                if not name:
                    st.warning("Isi nama pasien.")
                else:
                    with st.spinner('Menganalisis...'):
                        # Resize 150x150 sesuai training data
                        proc_img = ImageOps.fit(img, (150, 150), Image.Resampling.LANCZOS)
                        arr = np.asarray(proc_img) / 255.0
                        arr = np.expand_dims(arr, axis=0)
                        
                        pred = model.predict(arr)
                        score = tf.nn.softmax(pred[0])
                        idx = np.argmax(score)
                        
                        diag = class_names[idx]
                        conf = 100 * np.max(score)
                        
                        color = "#22c55e" if diag == "No Tumor" else "#ef4444"
                        st.markdown(f"""
                        <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;">
                            <h2 style="color:white !important;">{diag}</h2>
                            <p style="color:white !important;">Confidence Score: {conf:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.bar_chart(pd.DataFrame(pred[0]*100, index=class_names, columns=["%"]))
                        
                        pdf = create_clinical_pdf(name, diag, conf, file)
                        b64 = base64.b64encode(pdf).decode('utf-8')
                        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Laporan_{name}.pdf"><button style="width:100%; padding:10px; background:#1e293b; color:white; border-radius:8px; border:none; cursor:pointer;">📄 DOWNLOAD PDF</button></a>', unsafe_allow_html=True)
        else:
            st.info("Upload citra MRI untuk memulai.")

# ==========================================
# 4. ROUTER
# ==========================================

def main():
    with st.sidebar:
        st.markdown("## 🏥 Menu")
        page = st.radio("", ["NeuroScan System", "Developer Profile"], label_visibility="collapsed")
    
    if page == "NeuroScan System":
        show_system()
    else:
        show_portfolio()

if __name__ == "__main__":
    main()