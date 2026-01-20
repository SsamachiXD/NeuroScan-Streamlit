# ==============================================
# NeuroScan AI – ULTIMATE SaaS MODE
# ==============================================
# Features:
# - Top navigation bar
# - Dark / Light mode toggle
# - Landing page
# - Clinical dashboard layout
# - Micro animations
# - SaaS-style spacing & hierarchy
# ==============================================

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
# STATE
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
)

# ===============================
# THEME
# ===============================
def inject_css(theme="light"):
    if theme == "light":
        bg = "#f8fafc"
        card = "rgba(255,255,255,0.9)"
        text = "#020617"
    else:
        bg = "#020617"
        card = "rgba(15,23,42,0.9)"
        text = "#e5e7eb"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {text};
    }}

    .stApp {{
        background: {bg};
    }}

    .topbar {{
        position: sticky;
        top: 0;
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        padding: 18px 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 20px 40px rgba(79,70,229,0.35);
    }}

    .topbar h1 {{
        color: white;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0;
    }}

    .card {{
        background: {card};
        backdrop-filter: blur(10px);
        border-radius: 22px;
        padding: 28px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }}

    .badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
    }}

    .safe {{ background: #22c55e; color: white; }}
    .danger {{ background: #ef4444; color: white; }}

    .fade {{ animation: fadeIn 0.6s ease; }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

# ===============================
# MODEL
# ===============================
@st.cache_resource
def load_model():
    path = "VGG16_medium.h5"
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)

model = load_model()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ===============================
# PDF
# ===============================
def create_pdf(patient_name, diagnosis, confidence, img_obj):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "NeuroScan AI Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Patient: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Diagnosis: {diagnosis}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Date: {time.strftime('%d-%m-%Y %H:%M')} ", ln=True)
    pdf.ln(10)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(img_obj).convert("RGB")
            img.save(tmp.name)
            pdf.image(tmp.name, x=60, w=90)
    except:
        pass
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This AI system is a clinical decision support tool, not a medical diagnosis.")
    return pdf.output(dest='S').encode('latin-1')

# ===============================
# TOP BAR
# ===============================
def topbar():
    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown("<div class='topbar'><h1>NeuroScan AI</h1></div>", unsafe_allow_html=True)
    with col2:
        toggle = st.button("🌙" if st.session_state.theme == "light" else "☀️")
        if toggle:
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()

# ===============================
# PAGES
# ===============================
def landing_page():
    topbar()
    st.markdown("<div class='card fade'>", unsafe_allow_html=True)
    st.markdown("## Clinical-grade AI for Brain Tumor Detection")
    st.markdown("NeuroScan AI is a deep learning powered decision support system for radiology.")
    st.markdown("</div>", unsafe_allow_html=True)


def system_page():
    topbar()

    if model is None:
        st.error("Model not found.")
        return

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("<div class='card fade'>", unsafe_allow_html=True)
        st.markdown("### Patient Input")
        name = st.text_input("Patient Name")
        file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
        run = st.button("Run AI Diagnosis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card fade'>", unsafe_allow_html=True)
        st.markdown("### AI Output")

        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, use_container_width=True)

            if run:
                if not name:
                    st.warning("Enter patient name")
                else:
                    with st.spinner("Analyzing MRI..."):
                        time.sleep(1)
                        img_resized = ImageOps.fit(img, (150,150))
                        arr = np.asarray(img_resized) / 255.0
                        arr = np.expand_dims(arr, axis=0)

                        pred = model.predict(arr)
                        score = tf.nn.softmax(pred[0])
                        idx = np.argmax(score)
                        diagnosis = class_names[idx]
                        confidence = 100 * np.max(score)

                        status = "safe" if diagnosis == "No Tumor" else "danger"

                        st.markdown(f"<span class='badge {status}'>{diagnosis}</span>", unsafe_allow_html=True)
                        st.markdown(f"## {confidence:.2f}% Confidence")

                        df = pd.DataFrame({"Class": class_names, "Probability": pred[0] * 100})
                        st.bar_chart(df.set_index("Class"))

                        pdf = create_pdf(name, diagnosis, confidence, file)
                        b64 = base64.b64encode(pdf).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="NeuroScan_{name}.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("Upload MRI image to begin")

        st.markdown("</div>", unsafe_allow_html=True)


def profile_page():
    topbar()
    st.markdown("<div class='card fade'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,3])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=160)

    with col2:
        st.markdown("## Rahmat Ardiansyah")
        st.markdown("AI | Health-Tech | Computer Vision")
        st.markdown("Informatics student focusing on medical deep learning systems.")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ROUTER
# ===============================
def main():
    page = st.sidebar.radio("Navigation", ["Landing", "System", "Developer"])

    if page == "Landing":
        landing_page()
    elif page == "System":
        system_page()
    else:
        profile_page()

if __name__ == '__main__':
    main()
