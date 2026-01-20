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
# 1. KONFIGURASI HALAMAN & MODERN CSS
# ==========================================
st.set_page_config(
    page_title="NeuroScan AI | Rahmat Ardiansyah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS INJECTION: MERUBAH TOTAL TAMPILAN DEFAULT STREAMLIT
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background Utama: Clean Clinical Look */
    .stApp {
        background-color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Card Container Styling - Kunci Tampilan Modern */
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Custom Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Styling Tombol Utama */
    div.stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.4);
    }
    div.stButton > button:hover {
        background-color: #0284c7;
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.5);
        transform: translateY(-2px);
    }

    /* Alert Boxes Customization */
    div[data-baseweb="notification"] {
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Hapus elemen default yang mengganggu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC (OPTIMIZED)
# ==========================================

@st.cache_resource
def load_model_ai():
    # Error Handling agar aplikasi tidak crash jika model belum ada
    model_path = 'VGG16_medium.h5'
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

def create_clinical_pdf(patient_name, diagnosis, confidence, img_file):
    pdf = FPDF()
    pdf.add_page()
    
    # Modern Header
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
    
    # Patient Info Box
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, txt="DATA PASIEN", ln=True)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(50, 10, txt="Nama Pasien", border='B')
    pdf.cell(0, 10, txt=f": {patient_name}", border='B', ln=True)
    pdf.cell(50, 10, txt="Waktu Scan", border='B')
    pdf.cell(0, 10, txt=f": {time.strftime('%d-%m-%Y %H:%M WIB')}", border='B', ln=True)
    pdf.cell(50, 10, txt="ID Referensi", border='B')
    pdf.cell(0, 10, txt=f": REF-{int(time.time())}", border='B', ln=True)
    
    pdf.ln(10)
    
    # Result Section
    pdf.set_font("Arial", 'B', 14)
    if diagnosis == "No Tumor":
        pdf.set_text_color(34, 197, 94) # Green
    else:
        pdf.set_text_color(239, 68, 68) # Red
        
    pdf.cell(0, 15, txt=f"HASIL DIAGNOSIS: {diagnosis.upper()}", ln=True, border=1, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)
    pdf.ln(5)
    pdf.cell(0, 8, txt=f"Tingkat Kepercayaan Model: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, txt="Model AI: VGG16 (Convolutional Neural Network)", ln=True)
    
    # Disclaimer
    pdf.set_y(-40)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 5, txt="PENAFIAN: Dokumen ini dihasilkan oleh sistem kecerdasan buatan (AI) sebagai alat pendukung keputusan klinis. Hasil ini BUKAN diagnosis medis final. Wajib dikonsultasikan dengan dokter spesialis radiologi atau neurologi.")
    
    return pdf.output(dest='S').encode('latin-1')

# Load Resources
model = load_model_ai()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ==========================================
# 3. PAGE UI COMPONENTS
# ==========================================

def render_header():
    st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0;'>NeuroScan AI</h1>
            <p style='color: #64748b; font-size: 1.1rem;'>
                Clinical Decision Support System for Brain Tumor Detection
            </p>
            <div style='height: 4px; width: 60px; background-color: #0ea5e9; border-radius: 2px; margin-top: 10px;'></div>
        </div>
    """, unsafe_allow_html=True)

def show_portfolio():
    st.markdown("## 👨‍💻 Developer Profile")
    
    col_profile, col_details = st.columns([1, 3], gap="medium")
    
    with col_profile:
        # Ganti URL ini dengan foto profil asli Anda di hosting/GitHub
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
        st.markdown("""
            <div style='text-align: center; margin-top: 10px;'>
                <h3 style='margin:0;'>Rahmat Ardiansyah</h3>
                <p style='color:#64748b; font-size: 0.9em;'>AI Enthusiast</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_details:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#0ea5e9; margin-top:0;">Summary</h4>
            <p style="color:#334155; line-height: 1.6;">
                Mahasiswa tingkat akhir Teknik Informatika <b>Universitas Muhammadiyah Riau (UMRI)</b>. 
                Berfokus pada implementasi Deep Learning dalam bidang kesehatan (Health-Tech). 
                Memiliki passion dalam menerjemahkan data kompleks menjadi solusi yang dapat ditindaklanjuti.
            </p>
            <hr style="border-top: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                <div>📍 <b>Lokasi:</b> Pekanbaru, Riau</div>
                <div>🎓 <b>Status:</b> Mahasiswa (2022)</div>
                <div>💼 <b>Minat:</b> AI, Computer Vision, Python</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tech Stack Grid
    st.markdown("### 🛠️ Technology Stack")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.button("Python", disabled=True, use_container_width=True)
    with c2: st.button("TensorFlow", disabled=True, use_container_width=True)
    with c3: st.button("Streamlit", disabled=True, use_container_width=True)
    with c4: st.button("Computer Vision", disabled=True, use_container_width=True)

def show_system():
    render_header()
    
    if model is None:
        st.error("⚠️ **System Error:** File Model (VGG16_medium.h5) tidak ditemukan. Mohon upload file model ke direktori root.")
        return

    col_control, col_display = st.columns([1, 1.5], gap="large")

    # --- LEFT PANEL: CONTROL ---
    with col_control:
        st.markdown("### 1. Upload Data")
        with st.container(border=True):
            patient_name = st.text_input("Nama Pasien", placeholder="Masukkan nama lengkap...")
            uploaded_file = st.file_uploader("Upload Citra MRI", type=["jpg", "png", "jpeg"])
            
            if uploaded_file:
                st.info("✅ Citra berhasil dimuat. Siap analisis.")
            
            analyze_trigger = st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True)

        st.markdown("### 📝 Catatan Sistem")
        st.markdown("""
        <div style='background-color: #f1f5f9; padding: 15px; border-radius: 8px; font-size: 0.85rem; color: #475569;'>
        <ul>
            <li>Pastikan citra MRI jelas dan tidak buram.</li>
            <li>Format yang didukung: JPG, PNG.</li>
            <li>Sistem menggunakan model VGG16 dengan akurasi teruji.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- RIGHT PANEL: DISPLAY & RESULTS ---
    with col_display:
        st.markdown("### 2. Visualization & Results")
        
        # Placeholder jika belum ada data
        if not uploaded_file:
            st.markdown("""
            <div style='border: 2px dashed #cbd5e1; border-radius: 12px; padding: 40px; text-align: center; color: #94a3b8;'>
                <h3 style='color: #cbd5e1;'>Menunggu Input</h3>
                <p>Silakan upload citra MRI pada panel sebelah kiri untuk melihat hasil analisis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Tampilkan Gambar
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Citra Input (MRI Scan)", use_container_width=True)
            
            if analyze_trigger:
                if not patient_name:
                    st.warning("⚠️ Mohon isi Nama Pasien terlebih dahulu.")
                else:
                    with st.spinner('Sedang memproses neuron network...'):
                        time.sleep(1.5) # Efek dramatis loading
                        
                        # Preprocessing
                        img_processed = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
                        img_array = np.asarray(img_processed) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Prediction
                        prediction = model.predict(img_array)
                        score = tf.nn.softmax(prediction[0])
                        class_idx = np.argmax(score)
                        diagnosis = class_names[class_idx]
                        confidence = 100 * np.max(score)
                        
                        # --- RESULT CARD (Modern UI) ---
                        st.markdown("---")
                        
                        # Warna Status
                        status_color = "#22c55e" if diagnosis == "No Tumor" else "#ef4444" # Green vs Red
                        status_text_color = "#ffffff"
                        
                        st.markdown(f"""
                        <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                            <h2 style="color: {status_text_color}; margin:0;">{diagnosis}</h2>
                            <p style="color: {status_text_color}; margin:0; opacity: 0.9;">Confidence Score: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Grafik Probabilitas
                        st.markdown("#### Detail Probabilitas")
                        probs = prediction[0] * 100
                        df_chart = pd.DataFrame({"Kondisi": class_names, "Probabilitas": probs})
                        st.bar_chart(df_chart.set_index("Kondisi"), color="#0ea5e9")
                        
                        # Tombol Download PDF
                        pdf_bytes = create_clinical_pdf(patient_name, diagnosis, confidence, uploaded_file)
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="NeuroScan_Report_{patient_name}.pdf" style="text-decoration:none;">'
                        href += f'<div style="background-color:#1e293b; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; margin-top:10px;">📄 DOWNLOAD PDF REPORT</div></a>'
                        st.markdown(href, unsafe_allow_html=True)

# ==========================================
# 4. MAIN APP ROUTER
# ==========================================

def main():
    # Sidebar Navigation Custom
    with st.sidebar:
        st.markdown("## 🏥 Menu")
        page = st.radio(
            "",
            ["NeuroScan System", "Developer Profile"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption(f"© 2026 NeuroScan AI\nVer 1.0.2 - Beta")
        
        st.markdown("""
        <div style="background-color: #e0f2fe; padding: 10px; border-radius: 6px; margin-top: 20px;">
            <small style="color: #0369a1;"><b>Tips:</b> Gunakan citra MRI dengan kontras tinggi untuk hasil terbaik.</small>
        </div>
        """, unsafe_allow_html=True)

    if page == "NeuroScan System":
        show_system()
    else:
        show_portfolio()

if __name__ == "__main__":
    main()