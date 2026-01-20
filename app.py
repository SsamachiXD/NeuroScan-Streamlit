import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
import os
from datetime import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="NeuroScan AI | Rahmat Ardiansyah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODERN CSS (DARK THEME FIXED) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); /* Background Utama Gelap */
        color: #f8fafc;
    }

    /* Glassmorphism Cards (Versi Gelap Default) */
    .glass-card {
        background: rgba(15, 23, 42, 0.7); /* Gelap Transparan */
        backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Memaksa teks dalam card menjadi putih */
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4, 
    .glass-card p, .glass-card li, .glass-card span, .glass-card div {
        color: #f1f5f9 !important;
    }
    .glass-card strong { color: #818cf8 !important; }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem;
        line-height: 1.2;
    }

    /* Modern Tech Badges */
    .tech-badge-modern {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        padding: 8px 16px;
        margin: 6px 4px;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease forwards;
    }

    .tech-badge-modern:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.2);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Social Links */
    .social-link {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 12px 24px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        text-decoration: none;
        color: #f8fafc !important;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        margin: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .social-link:hover {
        transform: translateY(-3px);
        background: rgba(255,255,255,0.15);
        color: #818cf8 !important;
    }

    /* Profile Image */
    .profile-container {
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        animation: float 6s ease-in-out infinite;
        border: 4px solid rgba(255,255,255,0.1);
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    /* Stats Card (GELAP) */
    .stat-card {
        background: rgba(15, 23, 42, 0.8);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .stat-card:hover {
        transform: scale(1.05);
        border-color: rgba(255,255,255,0.3);
        background: rgba(30, 41, 59, 0.9);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label { color: #cbd5e1; font-weight: 600; }

    /* Result Card */
    .result-card {
        animation: slideIn 0.5s ease forwards;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Button Override */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white; border: none; padding: 0.75rem 2rem;
        font-weight: 700; border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }

    /* Input Styling (Agar tetap terang & mudah diisi) */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(255,255,255,0.2);
        padding: 0.75rem;
        background-color: rgba(255,255,255,0.9) !important;
        color: #0f172a !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #818cf8;
        background-color: #ffffff !important;
    }
    
    /* Sidebar Styling (GELAP) */
    [data-testid="stSidebar"] {
        background: #020617; /* Very Dark Blue */
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Memaksa teks sidebar jadi terang */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .section-header {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 1rem;
        position: relative;
        padding-left: 1rem;
        border-left: 4px solid #818cf8;
    }
    
    /* Info Box Dark Style */
    .info-box-dark {
        padding: 1.5rem; 
        background: rgba(15, 23, 42, 0.6); 
        border-radius: 16px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
    }
    .info-box-dark:hover {
        transform: translateY(-5px);
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255,255,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = 'VGG16_medium.h5'
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

model = load_model()
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def preprocess_image(image):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def create_pdf(patient_name, diagnosis, confidence, probabilities):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(190, 15, txt="NeuroScan AI", ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(190, 10, txt="Clinical Decision Support System", ln=True, align='C')
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, txt="LAPORAN DIAGNOSIS MEDIS", ln=True, align='C')
    
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(190, 10, txt=f"Nama Pasien: {patient_name}", ln=True)
    pdf.cell(190, 10, txt=f"Tanggal Pemeriksaan: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", ln=True)
    pdf.cell(190, 10, txt=f"NIM Pengembang: 220405010", ln=True)
    
    pdf.ln(5)
    pdf.set_fill_color(240, 253, 244) if diagnosis == 'No Tumor' else pdf.set_fill_color(254, 242, 242)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 15, txt=f"  Hasil Diagnosis: {diagnosis}", ln=True, fill=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(190, 10, txt=f"  Confidence Level: {confidence:.2f}%", ln=True, fill=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="Detail Probabilitas Klasifikasi:", ln=True)
    pdf.set_font("Arial", size=11)
    for cls, prob in probabilities.items():
        pdf.cell(190, 8, txt=f"  {cls}: {prob:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(190, 5, txt="Disclaimer: Hasil diagnosis ini merupakan prediksi dari sistem AI dan harus dikonfirmasi oleh tenaga medis profesional. NeuroScan AI dikembangkan sebagai alat bantu pendukung keputusan klinis.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. MODERN SIDEBAR (DARK) ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
        <h2 style="margin: 0; background: linear-gradient(135deg, #a5b4fc 0%, #e0e7ff 100%); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900;">
            NeuroScan AI
        </h2>
        <p style="color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;">
            Clinical Decision Support System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["🏠 Portfolio", "🔬 AI Dashboard", "📊 About Project"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px; margin-top: 1rem; border: 1px solid rgba(255,255,255,0.1);">
        <p style="font-size: 0.75rem; color: #cbd5e1; margin: 0; text-align: center;">
            <strong>Developer:</strong> Rahmat Ardiansyah<br>
            <strong>NIM:</strong> 220405010<br>
            <strong>Institution:</strong> UMRI
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. MODERN PORTFOLIO PAGE ---
if menu == "🏠 Portfolio":
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 2], gap="large")
    
    with col1:
        st.markdown('<div class="profile-container">', unsafe_allow_html=True)
        if os.path.exists("rahmat2.jpg"):
            st.image("rahmat2.jpg", use_container_width=True)
        elif os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", use_container_width=True)
        else:
            st.markdown("""
            <div style="aspect-ratio: 1; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                        display: flex; align-items: center; justify-content: center; border-radius: 24px;">
                <span style="font-size: 6rem;">👨‍💻</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats Cards (GELAP)
        st.markdown("<br>", unsafe_allow_html=True)
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">4</div>
                <div class="stat-label">Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with stat_col2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">95%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="animation: fadeInUp 0.8s ease;">
            <p style="color: #818cf8; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                👋 Hello, I'm
            </p>
            <h1 class="gradient-text">Rahmat Ardiansyah</h1>
            <p style="color: #cbd5e1; font-size: 1.25rem; line-height: 1.8; margin-top: 1rem;">
                Aku adalah mahasiswa <strong>Teknik Informatika</strong> di <strong>Universitas Muhammadiyah Riau</strong> 
                dengan passion dalam <strong>UI/UX Design</strong> dan <strong>Artificial Intelligence</strong>. 
                Menggabungkan estetika desain dengan kecerdasan buatan untuk menciptakan solusi inovatif.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Info Grid (GELAP)
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown("""
            <div class="info-box-dark">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📍</div>
                <div style="font-weight: 700; color: #f1f5f9;">Pekanbaru</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">Riau, Indonesia</div>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown("""
            <div class="info-box-dark">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎓</div>
                <div style="font-weight: 700; color: #f1f5f9;">UMRI</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">Angkatan 2022</div>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col3:
            st.markdown("""
            <div class="info-box-dark">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💼</div>
                <div style="font-weight: 700; color: #f1f5f9;">220405010</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">NIM</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🛠 Tech Stack & Expertise</div>', unsafe_allow_html=True)
        
        tech_stack = {
            "Languages": ["Python", "JavaScript", "C++"],
            "AI/ML": ["TensorFlow", "PyTorch", "Scikit-learn"],
            "Web Dev": ["Streamlit", "React", "Tailwind CSS"],
            "Design": ["Figma", "Adobe XD", "Framer"]
        }
        
        for category, techs in tech_stack.items():
            st.markdown(f"<p style='color: #cbd5e1; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>{category}</p>", unsafe_allow_html=True)
            badges_html = "".join([
                f'<span class="tech-badge-modern" style="animation-delay: {i*0.1}s;">{tech}</span>' 
                for i, tech in enumerate(techs)
            ])
            st.markdown(badges_html, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔗 Connect With Me</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
            <a href="https://github.com/SsamachiXD" target="_blank" class="social-link">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                GitHub
            </a>
            <a href="https://www.instagram.com/rahmatt.ah" target="_blank" class="social-link">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/>
                </svg>
                Instagram
            </a>
            <a href="https://www.tiktok.com/@rahmatt.ah" target="_blank" class="social-link">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/>
                </svg>
                TikTok
            </a>
        </div>
        """, unsafe_allow_html=True)

# --- 6. AI DASHBOARD ---
elif menu == "🔬 AI Dashboard":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 3rem;">NeuroScan AI Dashboard</h1>
        <p style="color: #cbd5e1; font-size: 1.1rem; max-width: 800px; margin: 1rem auto;">
            Sistem Pendukung Keputusan Klinis berbasis <strong>Deep Learning VGG16</strong> 
            untuk deteksi dini tumor otak melalui analisis citra MRI.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1.3], gap="large")
    
    with c_left:
        # Gunakan card transparan gelap
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # HEADER "INPUT DATA PASIEN" SUDAH DIHAPUS DISINI
        
        p_name = st.text_input("Nama Lengkap Pasien", placeholder="contoh: Dr. Budi Santoso")
        uploaded_file = st.file_uploader(
            "Upload Citra MRI Otak (150x150px)", 
            type=["jpg", "png", "jpeg"],
            help="Format: JPG, PNG, JPEG | Resolusi optimal: 150x150 piksel"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Preview Citra", use_container_width=True)
        
        agree = st.checkbox("✓ Saya menyetujui pemrosesan data medis untuk keperluan diagnosis")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Info Box
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
        **ℹ️ Informasi Sistem:**
        - Model: VGG16 CNN
        - Input: 150×150 RGB
        - Classes: 4 (Glioma, Meningioma, Pituitary, Normal)
        - Framework: TensorFlow 2.15.0
        """)

    with c_right:
        if predict_btn:
            if not model:
                st.error("⚠️ Model AI tidak ditemukan. Pastikan file `VGG16_medium.h5` tersedia.")
            elif not uploaded_file or not p_name or not agree:
                st.warning("⚠️ Mohon lengkapi semua field dan centang persetujuan data.")
            else:
                try:
                    with st.spinner("🧠 Menganalisis struktur neural network..."):
                        img = Image.open(uploaded_file).convert('RGB')
                        processed = preprocess_image(img)
                        
                        pred = model.predict(processed)
                        idx = np.argmax(pred[0])
                        label = CLASS_NAMES[idx]
                        conf = float(pred[0][idx] * 100)
                        probs = {k: float(v * 100) for k, v in zip(CLASS_NAMES, pred[0])}
                    
                    # Result Card
                    color_bg = "#ecfdf5" if label == "No Tumor" else "#fef2f2"
                    color_text = "#065f46" if label == "No Tumor" else "#991b1b"
                    emoji = "✅" if label == "No Tumor" else "⚠️"
                    
                    st.markdown(f"""
                    <div class="result-card" style="background: {color_bg}; border-left: 5px solid {color_text};">
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">{emoji}</div>
                            <h3 style="color: {color_text}; margin: 0;">Hasil Diagnosis</h3>
                            <h1 style="color: {color_text}; font-weight: 900; margin: 0.5rem 0;">{label}</h1>
                            <div style="background: white; padding: 1rem; border-radius: 12px; margin-top: 1rem; display: inline-block;">
                                <span style="color: #64748b; font-weight: 600;">Confidence Score:</span>
                                <span style="color: {color_text}; font-weight: 900; font-size: 1.5rem; margin-left: 0.5rem;">{conf:.2f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Probabilitas", "🖼️ Citra Input", "📄 Laporan"])
                    
                    with tab1:
                        st.markdown("#### Distribusi Probabilitas Klasifikasi")
                        
                        # Custom bar chart with colors
                        import pandas as pd
                        df_probs = pd.DataFrame({
                            'Kelas': list(probs.keys()),
                            'Probabilitas (%)': list(probs.values())
                        })
                        
                        st.bar_chart(df_probs.set_index('Kelas'), color="#667eea")
                        
                        # Detailed table
                        st.markdown("**Detail Numerik:**")
                        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                            bar_width = int(prob)
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                    <span style="font-weight: 600;">{cls}</span>
                                    <span style="color: #667eea; font-weight: 700;">{prob:.2f}%</span>
                                </div>
                                <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
                                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); 
                                                width: {bar_width}%; height: 100%; transition: width 0.5s ease;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab2:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(img, caption="Citra Original", use_container_width=True)
                        with col_img2:
                            st.markdown(f"""
                            **Metadata Citra:**
                            - **Filename:** {uploaded_file.name}
                            - **Size:** {uploaded_file.size / 1024:.2f} KB
                            - **Format:** {img.format}
                            - **Dimensions:** {img.size[0]}×{img.size[1]}
                            - **Mode:** {img.mode}
                            
                            **Preprocessing:**
                            - Resized: 150×150 px
                            - Normalized: [0, 1]
                            - Color: RGB
                            """)
                    
                    with tab3:
                        st.markdown("#### 📋 Laporan Medis Digital")
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid #e2e8f0;">
                            <h4 style="margin-top: 0;">EXECUTIVE SUMMARY</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;"><strong>Pasien</strong></td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">{p_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;"><strong>Tanggal</strong></td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">{datetime.now().strftime('%d %B %Y, %H:%M WIB')}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;"><strong>Diagnosis</strong></td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; color: {color_text}; font-weight: 700;">{label}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem;"><strong>Confidence</strong></td>
                                    <td style="padding: 0.5rem; font-weight: 700;">{conf:.2f}%</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        pdf_data = create_pdf(p_name, label, conf, probs)
                        st.download_button(
                            label="📥 Download Laporan PDF Lengkap",
                            data=pdf_data,
                            file_name=f"NeuroScan_Report_{p_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        st.caption("📌 Laporan ini dapat digunakan sebagai referensi konsultasi dengan tenaga medis profesional.")
                        
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.7); border-radius: 20px; backdrop-filter: blur(10px);">
                <div style="font-size: 4rem; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;">🧠</div>
                <h3 style="color: #64748b; font-weight: 600;">Siap Memulai Diagnosis</h3>
                <p style="color: #94a3b8;">Upload citra MRI dan klik tombol Analisis untuk memulai</p>
            </div>
            """, unsafe_allow_html=True)

# --- 7. ABOUT PROJECT PAGE ---
elif menu == "📊 About Project":
    st.markdown('<h1 class="gradient-text" style="text-align: center;">Tentang NeuroScan AI</h1>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Overview (GELAP)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Overview Proyek")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **NeuroScan AI** adalah Clinical Decision Support System (CDSS) berbasis web yang dikembangkan 
        untuk membantu deteksi dini tumor otak melalui analisis citra MRI menggunakan teknologi Deep Learning.
        
        **Tujuan Utama:**
        - Mempercepat proses screening awal
        - Memberikan second opinion untuk tenaga medis
        - Meningkatkan akurasi diagnosis
        - Aksesibilitas layanan kesehatan digital
        """)
    
    with col_info2:
        st.markdown("""
        **Spesifikasi Teknis:**
        - **Arsitektur:** VGG16 Convolutional Neural Network
        - **Input:** Citra MRI 150×150 RGB
        - **Output:** 4 Kelas (Glioma, Meningioma, Pituitary, Normal)
        - **Framework:** TensorFlow 2.15.0, Streamlit
        - **Deployment:** Streamlit Cloud
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technical Architecture (GELAP)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Arsitektur Sistem")
    
    st.markdown("""
    ```
    ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
    │  MRI Image      │ ───> │  Preprocessing   │ ───> │  VGG16 Model    │
    │  (User Upload)  │      │  (150×150 RGB)   │      │  (Deep Learning)│
    └─────────────────┘      └──────────────────┘      └─────────────────┘
                                                                │
                                                                ▼
    ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
    │  PDF Report     │ <─── │  Visualization   │ <─── │  Classification │
    │  (Download)     │      │  (Dashboard)     │      │  (4 Classes)    │
    └─────────────────┘      └──────────────────┘      └─────────────────┘
    ```
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Developer Info (GELAP)
    col_dev1, col_dev2, col_dev3 = st.columns(3)
    
    with col_dev1:
        st.markdown("""
        <div class="info-box-dark">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">👨‍💻</div>
            <h4>Developer</h4>
            <p style="font-weight: 700; color: #818cf8;">Rahmat Ardiansyah</p>
            <p style="color: #94a3b8; font-size: 0.875rem;">Teknik Informatika UMRI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_dev2:
        st.markdown("""
        <div class="info-box-dark">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎓</div>
            <h4>NIM</h4>
            <p style="font-weight: 700; color: #818cf8;">220405010</p>
            <p style="color: #94a3b8; font-size: 0.875rem;">Angkatan 2022</p>
        </div>
        """, unsafe_allow_html=True)

    with col_dev3:
        st.markdown("""
        <div class="info-box-dark">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📧</div>
            <h4>Contact</h4>
            <p style="font-weight: 700; color: #818cf8;">rahmat@student.umri.ac.id</p>
            <p style="color: #94a3b8; font-size: 0.875rem;">Pekanbaru, Riau</p>
        </div>
        """, unsafe_allow_html=True)