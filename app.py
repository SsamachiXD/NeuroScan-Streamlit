import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Rahmat.Dev | NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS CUSTOM (STYLE WEBSITE & LOGO SOSMED) ---
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* BACKGROUND PUTIH BERSIH */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* TYPOGRAPHY */
    .brand-text { color: #0ea5e9; font-weight: 800; }
    
    .hero-name {
        font-size: 3rem;
        font-weight: 900;
        color: #000000;
        line-height: 1.2;
        margin-bottom: 15px;
        letter-spacing: -1px;
    }
    
    .text-bio { 
        color: #0f172a !important; 
        font-weight: 500;
        font-size: 1.15rem;
        line-height: 1.6;
    }
    
    .text-desc { 
        color: #334155 !important;
        font-weight: 500;
        font-size: 1.05rem;
    }
    
    /* TECH BADGES */
    .tech-badge {
        display: inline-block;
        background-color: #e0f2fe; 
        color: #0369a1; 
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid #bae6fd;
    }

    /* SOCIAL ICONS HOVER EFFECT */
    .social-icon {
        transition: transform 0.2s;
    }
    .social-icon:hover {
        transform: scale(1.1);
    }

    /* CARD STYLE */
    .custom-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    /* BUTTON STYLE */
    div.stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 700;
        border-radius: 8px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #0284c7;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL & FUNGSI ---
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
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(190, 15, txt="NeuroScan AI - Medical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(190, 10, txt=f"Nama Pasien: {patient_name}", ln=True)
    pdf.cell(190, 10, txt=f"Tanggal: {st.session_state.get('date', 'Hari ini')}", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(240, 253, 244) if diagnosis == 'No Tumor' else pdf.set_fill_color(254, 242, 242)
    pdf.cell(190, 15, txt=f"  Hasil: {diagnosis} ({confidence:.2f}%)", ln=True, fill=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="Rincian Probabilitas:", ln=True)
    pdf.set_font("Arial", size=11)
    for cls, prob in probabilities.items():
        pdf.cell(190, 8, txt=f"- {cls}: {prob:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.markdown("### NeuroScan AI")
    st.markdown("**Rahmat Ardiansyah**")
    st.markdown("---")
    menu = st.radio("Navigasi", ["Portfolio Saya", "Dashboard AI"])
    st.markdown("---")
    st.caption("© 2026 Teknik Informatika UMRI")

# --- 5. HALAMAN PORTFOLIO ---
if menu == "Portfolio Saya":
    st.write("") 
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        if os.path.exists("rahmat2.jpg"):
            st.image("rahmat2.jpg", use_container_width=True)
        elif os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", use_container_width=True)
        else:
            st.info("Foto belum diupload")

    with col2:
        st.markdown('<span class="brand-text">Halo, Saya</span>', unsafe_allow_html=True)
        st.markdown('<div class="hero-name">Rahmat Ardiansyah</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="text-bio">
        Mahasiswa Teknik Informatika di <b>Universitas Muhammadiyah Riau</b> dengan ketertarikan mendalam 
        pada pengembangan <b>UI/UX Design</b> dan <b>Artificial Intelligence</b>. <br><br>
        Saya menggabungkan estetika desain dengan kecerdasan sistem untuk menciptakan solusi digital yang berdampak.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ") 
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📍 Domisili:** Pekanbaru, Riau")
        with c2:
            st.markdown("**🎓 Kampus:** UMRI (2022)")

        st.markdown("### 🛠 Tech Stack")
        badges = ["Python", "TensorFlow", "Streamlit", "Figma", "Tailwind", "C++", "React"]
        st.markdown(" ".join([f'<span class="tech-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔗 Connect with Me")
        
        # --- LOGO & LINKS SECTION ---
        st.markdown("""
        <div style="display: flex; gap: 20px; align-items: center; margin-top: 10px;">
            <a href="https://www.instagram.com/rahmatt.ah?igsh=eTh3Mjc5OG42bnZs" target="_blank" style="text-decoration: none;">
                <img class="social-icon" src="https://img.icons8.com/fluency/48/instagram-new.png" width="45" title="Instagram"/>
            </a>
            
            <a href="https://www.tiktok.com/@rahmatt.ah?_r=1&_t=ZS-93EgkA0aJyU" target="_blank" style="text-decoration: none;">
                <img class="social-icon" src="https://img.icons8.com/fluency/48/tiktok.png" width="45" title="TikTok"/>
            </a>
            
            <a href="https://github.com/SsamachiXD" target="_blank" style="text-decoration: none;">
                <img class="social-icon" src="https://img.icons8.com/fluency/48/github.png" width="45" title="GitHub"/>
            </a>
        </div>
        """, unsafe_allow_html=True)

# --- 6. HALAMAN DASHBOARD AI ---
elif menu == "Dashboard AI":
    st.markdown('<h1 style="color:#000;">🧠 NeuroScan <span style="color:#0ea5e9;">AI Dashboard</span></h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-desc">
    Sistem cerdas pendukung keputusan klinis berbasis <b>Deep Learning (VGG16)</b>. <br>
    Upload citra MRI otak untuk mendeteksi 4 kondisi: <i>Glioma, Meningioma, Pituitary,</i> atau <i>Normal</i>.
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.5], gap="medium")
    
    with left_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Data Pasien")
        p_name = st.text_input("Nama Lengkap", placeholder="Masukkan nama pasien...")
        uploaded_file = st.file_uploader("Upload MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])
        agree = st.checkbox("Saya menyetujui pemrosesan data medis.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("JALANKAN DIAGNOSIS", type="primary", use_container_width=True)
        if st.button("RESET SISTEM", use_container_width=True):
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with right_col:
        if btn_run:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            if not model:
                st.error("❌ Model 'VGG16_medium.h5' tidak ditemukan.")
            elif not uploaded_file or not p_name or not agree:
                st.warning("⚠️ Harap lengkapi semua data dan setujui konfirmasi.")
            else:
                try:
                    with st.spinner("Sedang menganalisis citra..."):
                        img = Image.open(uploaded_file).convert('RGB')
                        processed = preprocess_image(img)
                        pred = model.predict(processed)
                        idx = np.argmax(pred[0])
                        label = CLASS_NAMES[idx]
                        conf = float(pred[0][idx] * 100)
                        probs = {k: float(v * 100) for k, v in zip(CLASS_NAMES, pred[0])}
                    
                    bg_color = "#f0fdf4" if label == "No Tumor" else "#fef2f2"
                    text_color = "#15803d" if label == "No Tumor" else "#b91c1c"
                    
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid {text_color}; margin-bottom: 20px;">
                        <h4 style="margin:0; color: {text_color};">Hasil Diagnosis</h4>
                        <h2 style="margin: 5px 0; color: {text_color}; font-weight: 800; font-size: 2rem;">{label}</h2>
                        <p style="margin:0; font-weight: 600; color: {text_color};">Confidence: {conf:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["📊 Statistik", "🖼️ Citra Asli"])
                    with tab1:
                        st.bar_chart(probs, color="#0ea5e9")
                        pdf_data = create_pdf(p_name, label, conf, probs)
                        st.download_button("📄 Download Laporan PDF", data=pdf_data, file_name=f"Result_{p_name}.pdf", mime="application/pdf", use_container_width=True)
                    with tab2:
                        st.image(img, caption="Citra Input Pasien", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Silakan upload data di panel kiri untuk memulai.")