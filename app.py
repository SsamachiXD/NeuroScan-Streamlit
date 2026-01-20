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
# 1. KONFIGURASI HALAMAN & ULTRA MODERN CSS
# ==========================================
st.set_page_config(
    page_title="NeuroScan AI | Rahmat Ardiansyah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS INJECTION: MODERN MEDICAL INTERFACE WITH HIGH CONTRAST
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Font & Smooth Animations */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Main App Background: Elegant Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f8fafc 100%);
        background-attachment: fixed;
    }

    /* Sidebar Modern Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid #e2e8f0;
        box-shadow: 4px 0 12px rgba(0, 0, 0, 0.03);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1e293b !important;
        font-weight: 500;
    }

    /* High Contrast Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: 2.75rem !important;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* All Text Elements: High Contrast */
    p, span, div, li, label {
        color: #1e293b !important;
        line-height: 1.7;
    }

    /* Card Container with Glass Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        box-shadow: 0 12px 48px rgba(14, 165, 233, 0.15);
        transform: translateY(-2px);
    }

    /* Premium Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: #ffffff !important;
        border: none;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        box-shadow: 0 12px 32px rgba(14, 165, 233, 0.6);
        transform: translateY(-3px) scale(1.02);
    }

    div.stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* Disabled Button Styling */
    div.stButton > button:disabled {
        background: #cbd5e1 !important;
        color: #64748b !important;
        box-shadow: none;
        cursor: not-allowed;
    }

    /* Input Fields: High Contrast */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 1rem;
        color: #0f172a !important;
        transition: all 0.2s;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
    }

    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #0ea5e9;
        background: #f0f9ff;
    }

    /* Alert Boxes: Enhanced Contrast */
    div[data-baseweb="notification"] {
        border-radius: 12px;
        border-left: 4px solid;
        background: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }

    .stSuccess {
        border-left-color: #22c55e !important;
        background: #f0fdf4 !important;
    }

    .stError {
        border-left-color: #ef4444 !important;
        background: #fef2f2 !important;
    }

    .stWarning {
        border-left-color: #f59e0b !important;
        background: #fffbeb !important;
    }

    .stInfo {
        border-left-color: #0ea5e9 !important;
        background: #f0f9ff !important;
    }

    /* Radio Button Styling */
    div[role="radiogroup"] label {
        background: #ffffff;
        padding: 12px 20px;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin: 4px 0;
        transition: all 0.2s;
        color: #1e293b !important;
        font-weight: 500;
    }

    div[role="radiogroup"] label:hover {
        background: #f0f9ff;
        border-color: #0ea5e9;
    }

    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        border-color: #0284c7;
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Image Display: Professional Frame */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        border: 3px solid #ffffff;
    }

    /* Caption Styling */
    .stCaption {
        color: #64748b !important;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 8px;
    }

    /* Chart Styling */
    [data-testid="stVegaLiteChart"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #0ea5e9 !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0ea5e9, #0284c7);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #0284c7, #0369a1);
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC (OPTIMIZED)
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
        st.error(f"❌ Error loading model: {e}")
        return None

def create_clinical_pdf(patient_name, diagnosis, confidence, img_obj):
    pdf = FPDF()
    pdf.add_page()
    
    # Modern Header with Gradient Effect
    pdf.set_fill_color(14, 165, 233)
    pdf.rect(0, 0, 210, 50, 'F')
    
    pdf.set_y(15)
    pdf.set_font("Arial", 'B', 28)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, txt="NeuroScan AI", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(240, 249, 255)
    pdf.cell(0, 8, txt="Clinical Decision Support System", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, txt="Brain Tumor Detection Report", ln=True, align='C')
    pdf.ln(15)
    
    # Patient Information Section
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(15, pdf.get_y(), 180, 50, 'F')
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(20)
    pdf.cell(0, 10, txt="INFORMASI PASIEN", ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_x(20)
    pdf.cell(60, 8, txt="Nama Pasien")
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f": {patient_name}", ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_x(20)
    pdf.cell(60, 8, txt="Tanggal & Waktu")
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f": {time.strftime('%d %B %Y, %H:%M WIB')}", ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_x(20)
    pdf.cell(60, 8, txt="ID Referensi")
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f": REF-NEURO-{int(time.time())}", ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_x(20)
    pdf.cell(60, 8, txt="Model AI")
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=": VGG16 CNN Architecture", ln=True)
    
    pdf.ln(10)
    
    # MRI Image Display
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img = Image.open(img_obj).convert('RGB')
            img.save(tmp_file.name)
            
            # Center alignment
            img_width = 70
            x_position = (210 - img_width) / 2
            pdf.image(tmp_file.name, x=x_position, w=img_width)
            pdf.ln(5)
            
            # Image caption
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 5, txt="MRI Brain Scan - Input Image", ln=True, align='C')
            os.unlink(tmp_file.name)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(239, 68, 68)
        pdf.cell(0, 10, txt="[Image Processing Error]", ln=True, align='C')

    pdf.ln(10)

    # Diagnosis Result Section
    if diagnosis == "No Tumor":
        pdf.set_fill_color(34, 197, 94)
        status_text = "NEGATIF - TIDAK DITEMUKAN TUMOR"
    else:
        pdf.set_fill_color(239, 68, 68)
        status_text = f"POSITIF - {diagnosis.upper()}"
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, txt=status_text, ln=True, border=1, align='C', fill=True)
    
    pdf.ln(8)
    
    # Confidence Details
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.5)
    pdf.rect(15, pdf.get_y(), 180, 25, 'D')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(20)
    pdf.cell(0, 8, txt="TINGKAT KEPERCAYAAN MODEL", ln=True)
    
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(14, 165, 233)
    pdf.set_x(20)
    pdf.cell(0, 12, txt=f"{confidence:.2f}%", ln=True)
    
    pdf.ln(5)
    
    # Clinical Notes
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, txt="CATATAN KLINIS:", ln=True)
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(51, 65, 85)
    
    if diagnosis == "No Tumor":
        note = "Hasil analisis menunjukkan tidak ditemukan indikasi tumor pada citra MRI yang dianalisis. Namun, hasil ini tetap memerlukan verifikasi dari radiolog bersertifikat."
    else:
        note = f"Terdeteksi indikasi {diagnosis} dengan tingkat kepercayaan {confidence:.2f}%. Diperlukan tindak lanjut segera dari dokter spesialis untuk diagnosis definitif dan rencana perawatan."
    
    pdf.multi_cell(0, 6, txt=note)
    
    # Footer Disclaimer
    pdf.set_y(-45)
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(0, pdf.get_y(), 210, 45, 'F')
    
    pdf.set_y(-40)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(239, 68, 68)
    pdf.cell(0, 6, txt="⚠️ DISCLAIMER MEDIS", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, txt="Dokumen ini dihasilkan oleh sistem kecerdasan buatan (AI) sebagai alat pendukung keputusan klinis. Hasil ini BUKAN diagnosis medis final dan tidak dapat menggantikan penilaian profesional dari dokter spesialis radiologi atau neurologi. Pasien wajib berkonsultasi dengan tenaga medis bersertifikat untuk diagnosis definitif dan rencana perawatan yang tepat. NeuroScan AI tidak bertanggung jawab atas keputusan medis yang diambil berdasarkan hasil sistem ini tanpa verifikasi profesional.", align='C')
    
    pdf.set_y(-15)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(14, 165, 233)
    pdf.cell(0, 5, txt="Developed by: Rahmat Ardiansyah | Universitas Muhammadiyah Riau", ln=True, align='C')
    pdf.set_font("Arial", size=8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, txt="NeuroScan AI v1.0.3 | © 2026", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# Load Resources
model = load_model_ai()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ==========================================
# 3. PAGE UI COMPONENTS
# ==========================================

def render_header():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
            <div style='margin-bottom: 2rem;'>
                <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🧠 NeuroScan AI</h1>
                <p style='color: #475569; font-size: 1.2rem; font-weight: 500; margin: 0;'>
                    Clinical Decision Support System for Brain Tumor Detection
                </p>
                <div style='height: 4px; width: 80px; background: linear-gradient(90deg, #0ea5e9, #0284c7); border-radius: 2px; margin-top: 12px;'></div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #0ea5e9, #0284c7); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 8px 24px rgba(14, 165, 233, 0.3);'>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>VGG16</div>
                <div style='font-size: 0.85rem; color: #e0f2fe; font-weight: 600;'>Deep Learning Model</div>
            </div>
        """, unsafe_allow_html=True)

def show_portfolio():
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 2.5rem;'>👨‍💻 Developer Profile</h1>
            <p style='color: #64748b; font-size: 1.1rem;'>Artificial Intelligence Enthusiast</p>
            <div style='height: 4px; width: 60px; background: linear-gradient(90deg, #0ea5e9, #0284c7); border-radius: 2px; margin: 12px auto;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Profile Section
    col_profile, col_details = st.columns([1, 2], gap="large")
    
    with col_profile:
        # Load local photo
        if os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", use_container_width=True)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=250)
        
        st.markdown("""
            <div style='text-align: center; margin-top: 20px; background: linear-gradient(135deg, #0ea5e9, #0284c7); padding: 20px; border-radius: 12px;'>
                <h2 style='margin:0; color: #ffffff !important; font-size: 1.5rem;'>Rahmat Ardiansyah</h2>
                <p style='color: #e0f2fe !important; font-size: 1rem; margin-top: 8px; font-weight: 500;'>AI Engineer & Developer</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_details:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border: 1px solid #e2e8f0; border-radius: 16px; padding: 28px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);'>
            <h3 style="color:#0ea5e9 !important; margin-top:0; font-size: 1.5rem;">📋 Professional Summary</h3>
            <p style="color:#1e293b !important; line-height: 1.8; font-size: 1rem;">
                Mahasiswa tingkat akhir Program Studi <strong>Teknik Informatika</strong> di <strong>Universitas Muhammadiyah Riau (UMRI)</strong> dengan fokus pada implementasi <em>Deep Learning</em> dan <em>Computer Vision</em> dalam bidang kesehatan (<strong>Health-Tech</strong>). 
                <br><br>
                Memiliki passion yang kuat dalam menerjemahkan data kompleks menjadi solusi teknologi yang dapat memberikan dampak nyata bagi masyarakat, khususnya dalam sistem diagnosis medis berbasis AI.
            </p>
            
            <div style='height: 2px; background: linear-gradient(90deg, transparent, #cbd5e1, transparent); margin: 20px 0;'></div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px;">
                <div style='background: #f0f9ff; padding: 16px; border-radius: 10px; border-left: 4px solid #0ea5e9;'>
                    <div style='font-size: 1.5rem; margin-bottom: 4px;'>📍</div>
                    <div style='font-weight: 600; color: #0f172a !important; margin-bottom: 4px;'>Lokasi</div>
                    <div style='color: #475569 !important;'>Pekanbaru, Riau</div>
                </div>
                <div style='background: #f0fdf4; padding: 16px; border-radius: 10px; border-left: 4px solid #22c55e;'>
                    <div style='font-size: 1.5rem; margin-bottom: 4px;'>🎓</div>
                    <div style='font-weight: 600; color: #0f172a !important; margin-bottom: 4px;'>Institusi</div>
                    <div style='color: #475569 !important;'>UMRI - Angkatan 2022</div>
                </div>
                <div style='background: #fef3c7; padding: 16px; border-radius: 10px; border-left: 4px solid #f59e0b;'>
                    <div style='font-size: 1.5rem; margin-bottom: 4px;'>💼</div>
                    <div style='font-weight: 600; color: #0f172a !important; margin-bottom: 4px;'>Spesialisasi</div>
                    <div style='color: #475569 !important;'>AI & Computer Vision</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technology Stack Section
    st.markdown("""
        <h3 style='color: #0f172a !important; font-size: 1.75rem; margin-bottom: 1rem;'>🛠️ Technology Stack & Expertise</h3>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    tech_stack = [
        ("🐍 Python", "Programming Language"),
        ("🧠 TensorFlow", "Deep Learning Framework"),
        ("⚡ Streamlit", "Web Application Framework"),
        ("👁️ Computer Vision", "Image Processing & Analysis")
    ]
    
    for col, (tech, desc) in zip([c1, c2, c3, c4], tech_stack):
        with col:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #ffffff, #f8fafc); border: 2px solid #e2e8f0; padding: 20px; border-radius: 12px; text-align: center; transition: all 0.3s; height: 120px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='font-size: 1.75rem; margin-bottom: 8px;'>{tech.split()[0]}</div>
                    <div style='font-weight: 700; color: #0f172a !important; font-size: 0.95rem;'>{tech.split(' ', 1)[1]}</div>
                    <div style='font-size: 0.75rem; color: #64748b !important; margin-top: 4px;'>{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    # Project Highlights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='color: #0f172a !important; font-size: 1.75rem; margin-bottom: 1rem;'>🚀 Project Highlights</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border: 1px solid #e2e8f0; border-radius: 16px; padding: 28px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);'>
            <h4 style='color: #0ea5e9 !important; margin-top: 0;'>NeuroScan AI - Brain Tumor Detection System</h4>
            <ul style='color: #1e293b !important; line-height: 2; font-size: 1rem;'>
                <li><strong>Architecture:</strong> VGG16 Convolutional Neural Network</li>
                <li><strong>Dataset:</strong> Brain MRI Images (4 Classes Classification)</li>
                <li><strong>Features:</strong> Real-time tumor detection, clinical PDF report generation, high-accuracy predictions</li>
                <li><strong>Technology:</strong> TensorFlow, Keras, Streamlit, PIL, FPDF</li>
                <li><strong>Impact:</strong> Mendukung tenaga medis dalam diagnosis awal tumor otak dengan akurasi tinggi</li>
            </ul>
            
            <div style='background: #f0f9ff; padding: 16px; border-radius: 10px; border-left: 4px solid #0ea5e9; margin-top: 20px;'>
                <strong style='color: #0369a1 !important;'>🎯 Objective:</strong>
                <p style='color: #1e293b !important; margin: 8px 0 0 0;'>
                    Mengembangkan sistem AI yang dapat membantu dokter dalam mendeteksi tumor otak secara cepat dan akurat, 
                    sehingga mempercepat proses diagnosis dan meningkatkan kualitas layanan kesehatan.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_system():
    render_header()
    
    if model is None:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fee2e2, #fef2f2); border: 2px solid #ef4444; border-radius: 16px; padding: 32px; text-align: center; box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);'>
                <h2 style='color: #991b1b !important; margin: 0 0 12px 0;'>⚠️ System Error</h2>
                <p style='color: #7f1d1d !important; font-size: 1.1rem; font-weight: 500;'>
                    File Model (VGG16_medium.h5) tidak ditemukan.<br>
                    Pastikan file sudah di-upload pada folder yang sama dengan app.py.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Main Layout
    col_control, col_display = st.columns([1, 1.5], gap="large")

    # LEFT PANEL: CONTROL
    with col_control:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08); margin-bottom: 24px;'>
                <h3 style='color: #0f172a !important; margin-top: 0; font-size: 1.5rem;'>📝 Data Input</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            patient_name = st.text_input(
                "Nama Pasien",
                placeholder="Masukkan nama lengkap pasien...",
                help="Nama akan digunakan pada laporan PDF"
            )
            
            uploaded_file = st.file_uploader(
                "Upload Citra MRI",
                type=["jpg", "png", "jpeg"],
                help="Format: JPG, PNG, JPEG (Max 200MB)"
            )
            
            if uploaded_file:
                st.success("✅ Citra berhasil dimuat. Siap untuk dianalisis!", icon="✅")
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_trigger = st.button(
                "🚀 JALANKAN ANALISIS",
                type="primary",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # System Information
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: 2px solid #0ea5e9; border-radius: 16px; padding: 24px; box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2);'>
                <h4 style='color: #0369a1 !important; margin-top: 0;'>📋 Panduan Penggunaan</h4>
                <ul style='color: #1e293b !important; line-height: 1.8; font-size: 0.95rem; margin: 0; padding-left: 20px;'>
                    <li>Pastikan citra MRI memiliki <strong>kontras tinggi</strong> dan tidak buram</li>
                    <li>Format yang didukung: <strong>JPG, PNG, JPEG</strong></li>
                    <li>Sistem menggunakan model <strong>VGG16</strong> dengan akurasi teruji</li>
                    <li>Hasil analisis dapat diunduh dalam format <strong>PDF</strong></li>
                    <li>Waktu analisis: <strong>~2-3 detik</strong></li>
                </ul>
                
                <div style='height: 2px; background: linear-gradient(90deg, transparent, #0ea5e9, transparent); margin: 16px 0;'></div>
                
                <div style='background: rgba(255, 255, 255, 0.7); padding: 12px; border-radius: 8px;'>
                    <strong style='color: #0c4a6e !important;'>ℹ️ Model Information:</strong>
                    <div style='color: #1e293b !important; font-size: 0.85rem; margin-top: 8px;'>
                        <strong>Architecture:</strong> VGG16 CNN<br>
                        <strong>Classes:</strong> 4 (Glioma, Meningioma, No Tumor, Pituitary)<br>
                        <strong>Input Size:</strong> 150x150 RGB
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # RIGHT PANEL: DISPLAY & RESULTS
    with col_display:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08); margin-bottom: 24px;'>
                <h3 style='color: #0f172a !important; margin-top: 0; font-size: 1.5rem;'>📊 Visualization & Results</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if not uploaded_file:
            st.markdown("""
            <div style='border: 3px dashed #cbd5e1; border-radius: 16px; padding: 60px 40px; text-align: center; background: linear-gradient(135deg, #ffffff, #f8fafc);'>
                <div style='font-size: 4rem; margin-bottom: 16px; opacity: 0.3;'>🧠</div>
                <h3 style='color: #94a3b8 !important; margin-bottom: 12px;'>Menunggu Input Data</h3>
                <p style='color: #cbd5e1 !important; font-size: 1rem;'>Silakan upload citra MRI pada panel sebelah kiri untuk memulai analisis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Display Uploaded Image
            image = Image.open(uploaded_file).convert('RGB')
            
            st.markdown("""
                <div style='background: #ffffff; border: 2px solid #e2e8f0; border-radius: 16px; padding: 16px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);'>
                    <div style='font-weight: 600; color: #475569 !important; margin-bottom: 12px; font-size: 0.95rem;'>📸 MRI Brain Scan - Input Image</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
            
            if analyze_trigger:
                if not patient_name:
                    st.warning("⚠️ Mohon isi Nama Pasien terlebih dahulu.", icon="⚠️")
                else:
                    with st.spinner('🔬 Sedang memproses neural network... Mohon tunggu.'):
                        time.sleep(2)
                        
                        # Image Preprocessing
                        target_size = (150, 150)
                        img_processed = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
                        img_array = np.asarray(img_processed) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Model Prediction
                        prediction = model.predict(img_array, verbose=0)
                        score = tf.nn.softmax(prediction[0])
                        class_idx = np.argmax(score)
                        diagnosis = class_names[class_idx]
                        confidence = 100 * np.max(score)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Result Card
                        if diagnosis == "No Tumor":
                            status_color = "linear-gradient(135deg, #22c55e, #16a34a)"
                            icon = "✅"
                            border_color = "#22c55e"
                        else:
                            status_color = "linear-gradient(135deg, #ef4444, #dc2626)"
                            icon = "⚠️"
                            border_color = "#ef4444"
                        
                        st.markdown(f"""
                        <div style="background: {status_color}; padding: 32px; border-radius: 16px; margin-bottom: 24px; text-align: center; box-shadow: 0 12px 48px rgba(0, 0, 0, 0.2); border: 3px solid {border_color};">
                            <div style='font-size: 3rem; margin-bottom: 8px;'>{icon}</div>
                            <h2 style="color: #ffffff !important; margin:0 0 12px 0; font-size: 2rem;">{diagnosis}</h2>
                            <div style='height: 2px; width: 60px; background: rgba(255, 255, 255, 0.5); margin: 0 auto 16px auto;'></div>
                            <p style="color: #ffffff !important; margin:0; font-size: 1.3rem; font-weight: 600;">
                                Confidence Score: {confidence:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability Chart
                        st.markdown("""
                            <div style='background: #ffffff; border: 2px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); margin-bottom: 24px;'>
                                <h4 style='color: #0f172a !important; margin-top: 0;'>📈 Detail Probabilitas per Kelas</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        probs = prediction[0] * 100
                        df_chart = pd.DataFrame({
                            "Kondisi": class_names,
                            "Probabilitas (%)": probs
                        })
                        
                        st.bar_chart(df_chart.set_index("Kondisi"), color="#0ea5e9", height=300)
                        
                        # Detailed Probability Table
                        st.markdown("""
                            <div style='background: #ffffff; border: 2px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); margin-bottom: 24px;'>
                                <h4 style='color: #0f172a !important; margin-top: 0;'>📋 Tabel Probabilitas Detail</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
                            if i == class_idx:
                                st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #0ea5e9, #0284c7); padding: 12px 20px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
                                        <span style='color: #ffffff !important; font-weight: 700;'>🎯 {class_name}</span>
                                        <span style='color: #ffffff !important; font-weight: 700; font-size: 1.1rem;'>{prob:.2f}%</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='background: #f8fafc; padding: 12px 20px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #e2e8f0;'>
                                        <span style='color: #475569 !important; font-weight: 500;'>{class_name}</span>
                                        <span style='color: #64748b !important; font-weight: 500;'>{prob:.2f}%</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Generate PDF
                        pdf_bytes = create_clinical_pdf(patient_name, diagnosis, confidence, uploaded_file)
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        
                        st.markdown(f"""
                            <a href="data:application/octet-stream;base64,{b64_pdf}" download="NeuroScan_Report_{patient_name.replace(' ', '_')}.pdf" style="text-decoration:none;">
                                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); color: #ffffff !important; padding: 20px; border-radius: 12px; text-align: center; font-weight: 700; font-size: 1.1rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); transition: all 0.3s; cursor: pointer; border: 2px solid #334155;">
                                    📄 DOWNLOAD LAPORAN PDF
                                </div>
                            </a>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            <div style='background: #fef3c7; border: 2px solid #f59e0b; border-radius: 12px; padding: 16px; margin-top: 24px;'>
                                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                                    <span style='font-size: 1.5rem; margin-right: 12px;'>⚕️</span>
                                    <strong style='color: #92400e !important; font-size: 1.05rem;'>Catatan Penting</strong>
                                </div>
                                <p style='color: #78350f !important; margin: 0; line-height: 1.6; font-size: 0.95rem;'>
                                    Hasil analisis ini merupakan <strong>pendukung keputusan klinis</strong> dan bukan diagnosis final. 
                                    Wajib dikonsultasikan dengan <strong>dokter spesialis radiologi atau neurologi</strong> untuk tindak lanjut medis yang tepat.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

# ==========================================
# 4. MAIN APP ROUTER
# ==========================================

def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #0ea5e9, #0284c7); border-radius: 12px; margin-bottom: 24px;'>
                <h2 style='color: #ffffff !important; margin: 0; font-size: 1.5rem;'>🏥 Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Pilih Halaman",
            ["🧠 NeuroScan System", "👨‍💻 Developer Profile"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff, #e0f2fe); padding: 20px; border-radius: 12px; border: 2px solid #0ea5e9; box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2);">
            <div style='text-align: center; margin-bottom: 12px;'>
                <strong style="color: #0369a1 !important; font-size: 1.1rem;">💡 Tips Penggunaan</strong>
            </div>
            <p style="color: #1e293b !important; margin: 0; font-size: 0.9rem; line-height: 1.6;">
                Gunakan citra MRI dengan <strong>kontras tinggi</strong> dan <strong>pencahayaan optimal</strong> untuk hasil analisis terbaik.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
            <div style='text-align: center; color: #64748b !important;'>
                <div style='font-weight: 700; font-size: 1rem; color: #0f172a !important; margin-bottom: 8px;'>NeuroScan AI</div>
                <div style='font-size: 0.85rem;'>Version 1.0.3 - Beta</div>
                <div style='font-size: 0.8rem; margin-top: 8px;'>© 2026 All Rights Reserved</div>
                <div style='height: 2px; width: 40px; background: linear-gradient(90deg, transparent, #0ea5e9, transparent); margin: 12px auto;'></div>
                <div style='font-size: 0.75rem; margin-top: 8px;'>Developed by<br><strong style='color: #0ea5e9 !important;'>Rahmat Ardiansyah</strong></div>
            </div>
        """, unsafe_allow_html=True)

    # Page Routing
    if "NeuroScan System" in page:
        show_system()
    else:
        show_portfolio()

if __name__ == "__main__":
    main()
