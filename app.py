import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
import base64
import os

# --- 1. CONFIG & LAYOUT ---
st.set_page_config(
    page_title="NeuroScan AI | Rahmat.Dev",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (MENIRU STYLE WEBSITE ANDA) ---
st.markdown("""
    <style>
    /* Import Font mirip Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Warna Brand: #0ea5e9 (Sky 500) */
    .brand-text { color: #0ea5e9; font-weight: 700; }
    .text-slate { color: #334155; }
    .text-light { color: #64748b; }
    
    /* Styling Header Portfolio */
    .hero-name {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 10px;
    }
    
    /* Styling Tech Stack Badges */
    .tech-badge {
        display: inline-block;
        background-color: #e0f2fe; /* Sky 100 */
        color: #0284c7; /* Sky 600 */
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #bae6fd;
    }

    /* Card Style untuk Dashboard */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Button Styling Override */
    div.stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0284c7;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC & MODEL LOADING ---
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
    # PENTING: Ukuran 150x150 sesuai model Anda
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
    pdf.cell(190, 15, txt="NeuroScan AI Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(190, 10, txt=f"Nama Pasien: {patient_name}", ln=True)
    pdf.cell(190, 10, txt=f"Tanggal: {st.session_state.get('date', 'Hari ini')}", ln=True)
    
    pdf.ln(5)
    pdf.set_fill_color(240, 253, 244) if diagnosis == 'No Tumor' else pdf.set_fill_color(254, 242, 242)
    pdf.cell(190, 15, txt=f"  Hasil: {diagnosis} ({confidence:.2f}%)", ln=True, fill=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="Detail Probabilitas:", ln=True)
    pdf.set_font("Arial", size=11)
    for cls, prob in probabilities.items():
        pdf.cell(190, 8, txt=f"- {cls}: {prob:.2f}%", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

# --- 4. SIDEBAR MENU ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.markdown("### NeuroScan AI")
    st.markdown("Developed by **Rahmat Ardiansyah**")
    st.markdown("---")
    
    menu = st.radio("Navigasi", ["Portfolio Saya", "Dashboard AI"])
    
    st.markdown("---")
    st.caption("© 2026 Teknik Informatika UMRI")

# --- 5. HALAMAN PORTFOLIO (Mirip Website) ---
if menu == "Portfolio Saya":
    # Spacer
    st.write("") 
    
    col1, col2 = st.columns([1, 1.8], gap="large")
    
    with col1:
        # Foto Profil dengan style rounded
        if os.path.exists("rahmat2.jpg"):
            st.image("rahmat2.jpg", use_container_width=True)
        elif os.path.exists("rahmat1.png"):
            st.image("rahmat1.png", use_container_width=True)
        else:
            st.warning("Upload foto profil (rahmat2.jpg) ke GitHub.")

    with col2:
        st.markdown('<span class="brand-text">Halo, Saya</span>', unsafe_allow_html=True)
        st.markdown('<div class="hero-name">Rahmat Ardiansyah</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <p class="text-slate" style="font-size: 1.1rem; line-height: 1.6;">
        Mahasiswa Teknik Informatika di <b>Universitas Muhammadiyah Riau</b> dengan ketertarikan mendalam 
        pada pengembangan <b>UI/UX Design</b> dan <b>Artificial Intelligence</b>. 
        Saya menggabungkan estetika desain dengan kecerdasan sistem.
        </p>
        """, unsafe_allow_html=True)
        
        # Info Grid
        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.markdown("📍 **Domisili**")
            st.markdown("Pekanbaru, Riau")
        with c_info2:
            st.markdown("🎓 **Universitas**")
            st.markdown("UMRI (Angkatan 2022)")

        st.markdown("### 🛠 Tech Stack & Tools")
        tech_list = ["Python", "TensorFlow", "Streamlit", "Figma", "Tailwind CSS", "C++", "JavaScript", "React"]
        
        # Render Badges
        badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in tech_list])
        st.markdown(badges_html, unsafe_allow_html=True)
        
        st.markdown("---")
        # Social Links (Text based mimics icons)
        st.markdown("""
        **Connect with me:** [GitHub](https://github.com/SsamachiXD) &nbsp;•&nbsp; [Instagram](https://instagram.com) &nbsp;•&nbsp; [LinkedIn](https://linkedin.com)
        """)

# --- 6. HALAMAN AI DASHBOARD (Fungsionalitas Utama) ---
elif menu == "Dashboard AI":
    st.markdown('<h1 style="color:#0f172a;">🧠 NeuroScan <span style="color:#0ea5e9;">AI Dashboard</span></h1>', unsafe_allow_html=True)
    st.markdown("""
    <p class="text-light">
    Sistem cerdas pendukung keputusan klinis berbasis <b>Deep Learning (VGG16)</b>. 
    Upload citra MRI otak untuk mendeteksi 4 kondisi: <i>Glioma, Meningioma, Pituitary,</i> atau <i>Normal</i>.
    </p>
    """, unsafe_allow_html=True)
    
    # Layout Split: Input Kiri, Output Kanan
    c_left, c_right = st.columns([1, 1.5], gap="medium")
    
    with c_left:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload Data Pasien")
        
        p_name = st.text_input("Nama Pasien", placeholder="Cth: Budi Santoso")
        uploaded_file = st.file_uploader("Upload Citra MRI", type=["jpg", "png", "jpeg"])
        
        agree = st.checkbox("Saya menyetujui pemrosesan data medis ini.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 JALANKAN DIAGNOSIS", type="primary", use_container_width=True)
        
        if st.button("🔄 Reset Sistem", use_container_width=True):
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        if predict_btn:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            if not model:
                st.error("Model VGG16 tidak ditemukan. Pastikan file .h5 sudah diupload.")
            elif not uploaded_file or not p_name or not agree:
                st.warning("⚠️ Mohon lengkapi nama, file gambar, dan persetujuan.")
            else:
                try:
                    with st.spinner("Sedang menganalisis struktur otak..."):
                        img = Image.open(uploaded_file).convert('RGB')
                        processed = preprocess_image(img)
                        
                        # Prediksi
                        pred = model.predict(processed)
                        idx = np.argmax(pred[0])
                        label = CLASS_NAMES[idx]
                        conf = float(pred[0][idx] * 100)
                        probs = {k: float(v * 100) for k, v in zip(CLASS_NAMES, pred[0])}
                    
                    # Tampilan Hasil
                    color_bg = "#f0fdf4" if label == "No Tumor" else "#fef2f2"
                    color_text = "#166534" if label == "No Tumor" else "#991b1b"
                    
                    st.markdown(f"""
                    <div style="background-color: {color_bg}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {color_text};">
                        <h4 style="margin:0; color: {color_text};">Hasil Diagnosis AI</h4>
                        <h2 style="margin: 5px 0; color: {color_text}; font-weight: 800;">{label}</h2>
                        <p style="margin:0; font-weight: 600;">Confidence: {conf:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tabs untuk detail
                    tab1, tab2 = st.tabs(["📊 Statistik", "🖼️ Citra Input"])
                    
                    with tab1:
                        st.caption("Probabilitas per Kelas:")
                        st.bar_chart(probs, color="#0ea5e9")
                        
                        # PDF Download
                        pdf_data = create_pdf(p_name, label, conf, probs)
                        st.download_button(
                            label="📄 Download Laporan PDF Resmi",
                            data=pdf_data,
                            file_name=f"Report_{p_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    with tab2:
                        st.image(img, width=250, caption=f"MRI: {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Placeholder State (Saat belum ada aksi)
            st.info("👈 Silakan isi formulir di panel kiri untuk memulai diagnosis.")
            st.markdown("""
            <div style="text-align: center; opacity: 0.5;">
                <img src="https://cdn-icons-png.flaticon.com/512/2814/2814781.png" width="100">
                <p>Menunggu input data...</p>
            </div>
            """, unsafe_allow_html=True)