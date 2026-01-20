# 🧠 NeuroScan AI: Clinical Decision Support System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)

**NeuroScan AI** adalah aplikasi berbasis web yang mengintegrasikan Deep Learning (arsitektur VGG16) untuk membantu tenaga medis dalam mendeteksi dan mengklasifikasikan jenis tumor otak melalui citra MRI. Aplikasi ini memisahkan antara profil profesional pengembang dan sistem diagnosis klinis untuk pengalaman pengguna yang lebih terstruktur.

---

### 🌟 Fitur Unggulan
* **Analisis Citra MRI:** Mengklasifikasikan 4 kondisi: *Glioma*, *Meningioma*, *Pituitary*, dan *No Tumor*.
* **Arsitektur VGG16:** Menggunakan model Convolutional Neural Network (CNN) yang telah dioptimasi untuk akurasi tinggi pada citra medis.
* **Laporan Klinis PDF:** Menghasilkan laporan otomatis yang berisi data pasien, hasil diagnosis, dan tingkat kepercayaan AI (Confidence Score).
* **Dual Interface:** Navigasi yang memisahkan antara Dashboard AI dan Portofolio Developer.

---

### 🌐 Live Demo
Aplikasi ini sudah dideploy dan dapat diakses sepenuhnya (termasuk fitur AI) melalui tautan berikut:
👉 **[Buka NeuroScan AI di Streamlit Cloud](https://deteksi-tumor-mri.streamlit.app/)** 

---

### 📂 Struktur Proyek
- `app.py`: File utama aplikasi (Streamlit UI & Logic).
- `requirements.txt`: Daftar pustaka (library) yang dibutuhkan oleh server.
- `VGG16_medium.h5`: Model Deep Learning yang telah dilatih (Pre-trained model).

---

### 💻 Cara Instalasi Lokal (Development)

Jika ingin menjalankan aplikasi ini di komputer sendiri:

1. **Clone Repository**
   ```bash
   git clone [https://github.com/SsamachiXD/NeuroScan-Streamlit.git](https://github.com/SsamachiXD/NeuroScan-Streamlit.git)
   cd NeuroScan-Streamlit