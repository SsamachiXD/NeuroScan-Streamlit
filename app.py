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

# CSS INJECTION: MERUBAH TOTAL TAMPILAN
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp { background-color: #f8fafc; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Card Container Styling */
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
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

    /* Hide Default Elements */
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
    # Pastikan nama file ini SAMA PERSIS dengan yang ada di folder GitHub Anda
    model_path = 'VGG16_medium.h5' 
    
    if not os.path.exists(model_path):
        return None
    try:
        # Load model tanpa compile agar lebih kompatibel
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt="DATA PASIEN", ln=True)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(50, 10, txt="Nama Pasien", border='B')
    pdf.cell(0, 10, txt=f": {patient_name}", border='B', ln=True)
    pdf.cell(50, 10, txt="Waktu Scan", border='B')
    pdf.cell(0, 10, txt=f": {time.strftime('%d-%m-%Y %H:%M WIB')}", border='B', ln=True)
    
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
    col1, col2 = st.columns([1, 3], gap="medium")
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0;">Rahmat Ardiansyah</h3>
            <p>Mahasiswa Teknik Informatika <b>Universitas Muhammadiyah Riau (UMRI)</b>.</p>
            <p>Fokus pada implementasi Deep Learning & Computer Vision.</p>
        </div>
        """, unsafe_allow_html=True)

def show_system():
    render_header()
    
    if model is None:
        st.error("⚠️ **System Error:** File Model (VGG16_medium.h5) tidak ditemukan. Mohon cek kembali nama file di GitHub Anda.")
        return

    col_control, col_display = st.columns([1, 1.5], gap="large")

    with col_control:
        st.markdown("### 1. Upload Data")
        with st.container(border=True):
            patient_name = st.text_input("Nama Pasien", placeholder="Masukkan nama...")
            uploaded_file = st.file_uploader("Upload Citra MRI", type=["jpg", "png", "jpeg"])
            
            analyze_trigger = st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True)

    with col_display:
        st.markdown("### 2. Visualization & Results")
        
        if not uploaded_file:
            st.info("Silakan upload citra MRI pada panel kiri.")
        else:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Citra Input", use_container_width=True)
            
            if analyze_trigger:
                if not patient_name:
                    st.warning("⚠️ Mohon isi Nama Pasien.")
                else:
                    with st.spinner('Sedang memproses...'):
                        time.sleep(1.0)
                        
                        # --- PERBAIKAN UTAMA DI SINI ---
                        # Mengubah ukuran ke 224x224 (Standar VGG16)
                        target_size = (224, 224)
                        img_processed = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
                        
                        img_array = np.asarray(img_processed)
                        img_array = img_array / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Prediksi
                        try:
                            prediction = model.predict(img_array)
                            score = tf.nn.softmax(prediction[0])
                            class_idx = np.argmax(score)
                            diagnosis = class_names[class_idx]
                            confidence = 100 * np.max(score)
                            
                            # Tampilkan Hasil
                            st.markdown("---")
                            color = "#22c55e" if diagnosis == "No Tumor" else "#ef4444"
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                                <h2 style="margin:0; color: white;">{diagnosis}</h2>
                                <p style="margin:0;">Confidence: {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Chart
                            df_chart = pd.DataFrame({"Kondisi": class_names, "Probabilitas": prediction[0]*100})
                            st.bar_chart(df_chart.set_index("Kondisi"), color="#0ea5e9")
                            
                            # PDF
                            pdf_bytes = create_clinical_pdf(patient_name, diagnosis, confidence, uploaded_file)
                            b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="Hasil_{patient_name}.pdf"><button style="width:100%; padding:10px; margin-top:10px; background:#1e293b; color:white; border:none; border-radius:5px;">📄 Download PDF</button></a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat prediksi: {e}")
                            st.info("Tips: Coba latih ulang model dengan ukuran gambar yang konsisten (224x224).")

# ==========================================
# 4. MAIN APP ROUTER
# ==========================================

def main():
    with st.sidebar:
        st.markdown("## 🏥 Menu")
        page = st.radio("", ["NeuroScan System", "Developer Profile"], index=0)
        st.markdown("---")
        st.caption("© 2026 NeuroScan AI")

    if page == "NeuroScan System":
        show_system()
    else:
        show_portfolio()

if __name__ == "__main__":
    main()