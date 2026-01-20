# 🧠 NeuroScan AI - Brain Tumor Detection System

![Version](https://img.shields.io/badge/version-1.0.3-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red)

## 📋 Deskripsi

**NeuroScan AI** adalah sistem Clinical Decision Support berbasis Deep Learning untuk deteksi tumor otak otomatis menggunakan citra MRI. Aplikasi ini dibangun menggunakan arsitektur **VGG16 Convolutional Neural Network** dan interface web modern dengan **Streamlit**.

### ✨ Fitur Utama

- 🔬 **Deteksi Otomatis**: Klasifikasi 4 jenis kondisi (Glioma, Meningioma, No Tumor, Pituitary)
- 📊 **Visualisasi Real-time**: Grafik probabilitas dan confidence score
- 📄 **Laporan PDF**: Generate laporan klinis profesional secara otomatis
- 🎨 **UI Modern**: Desain clean dengan high contrast untuk readability optimal
- ⚡ **Performa Tinggi**: Analisis dalam 2-3 detik
- 📱 **Responsive Design**: Tampilan optimal di desktop dan mobile

## 🚀 Quick Start

### Prasyarat

- Python 3.9+
- pip package manager
- 4GB+ RAM (untuk model VGG16)

### Instalasi

1. **Clone atau Download Repository**
```bash
cd /app
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Pastikan File Model Ada**
Pastikan file `VGG16_medium.h5` berada di folder yang sama dengan `app.py`

4. **Jalankan Aplikasi**
```bash
streamlit run app.py --server.port=8501
```

5. **Akses Aplikasi**
Buka browser dan akses: `http://localhost:8501`

## 📁 Struktur File

```
/app/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Python dependencies
├── VGG16_medium.h5        # Pre-trained model (61MB)
├── rahmat1.png            # Foto profile developer
├── .streamlit/
│   └── config.toml        # Konfigurasi Streamlit
└── README.md              # Dokumentasi
```

## 🛠️ Technology Stack

| Teknologi | Versi | Fungsi |
|-----------|-------|--------|
| **Python** | 3.9+ | Programming language |
| **TensorFlow** | 2.15.0 | Deep learning framework |
| **Streamlit** | 1.31.0 | Web application framework |
| **NumPy** | 1.24.3 | Numerical computation |
| **Pandas** | 2.1.4 | Data manipulation |
| **PIL (Pillow)** | 10.2.0 | Image processing |
| **FPDF** | 1.7.2 | PDF generation |
| **H5py** | 3.10.0 | HDF5 file handling |

## 🧬 Model Information

- **Architecture**: VGG16 (16-layer CNN)
- **Input Size**: 150x150 RGB
- **Output Classes**: 4 classes
  1. Glioma Tumor
  2. Meningioma Tumor
  3. No Tumor
  4. Pituitary Tumor
- **Model Size**: ~61 MB
- **Inference Time**: ~2-3 seconds

## 💡 Cara Penggunaan

### 1. Upload Citra MRI
- Klik tombol "Upload Citra MRI" di panel kiri
- Pilih file gambar (JPG, PNG, JPEG)
- Pastikan citra memiliki kontras tinggi

### 2. Input Data Pasien
- Isi nama pasien pada field "Nama Pasien"
- Nama ini akan digunakan pada laporan PDF

### 3. Jalankan Analisis
- Klik tombol "🚀 JALANKAN ANALISIS"
- Tunggu 2-3 detik untuk proses neural network
- Hasil akan ditampilkan di panel kanan

### 4. Review Hasil
- Lihat diagnosis dan confidence score
- Cek detail probabilitas per kelas
- Review grafik visualisasi

### 5. Download Laporan
- Klik tombol "📄 DOWNLOAD LAPORAN PDF"
- File PDF akan otomatis terunduh
- Laporan berisi informasi lengkap analisis

## 🎨 Perbaikan Tampilan (v1.0.3)

### ✅ Perubahan Mayor

1. **High Contrast Design**
   - Text color: `#0f172a` (hampir hitam) untuk readability maksimal
   - Background: Gradient `#f0f9ff` → `#e0f2fe` → `#f8fafc`
   - Tidak ada lagi text yang menyatu dengan background

2. **Modern UI Components**
   - Glass morphism effect pada cards
   - Smooth transitions dan hover effects
   - Professional gradient buttons
   - Enhanced shadows dan borders

3. **Clinical Blue Theme**
   - Primary: `#0ea5e9` (Cyan 500)
   - Secondary: `#0284c7` (Cyan 600)
   - Accent: `#0369a1` (Cyan 700)
   - Memberikan kesan medis, bersih, dan terpercaya

4. **Typography Enhancement**
   - Font: Inter (modern sans-serif)
   - Clear font weights dan sizes
   - Optimal line-height untuk readability

5. **Responsive Layout**
   - Grid system untuk berbagai screen sizes
   - Optimasi spacing dan padding
   - Mobile-friendly design

## 📊 Performa

- **Load Time**: < 3 detik
- **Inference Time**: 2-3 detik per image
- **Memory Usage**: ~2GB (termasuk model)
- **Supported Image Formats**: JPG, PNG, JPEG
- **Max Image Size**: 200MB

## ⚠️ Disclaimer Medis

> **PENTING**: Hasil analisis dari sistem ini merupakan **alat pendukung keputusan klinis** dan bukan diagnosis medis final. Hasil ini:
> 
> - ❌ Tidak dapat menggantikan penilaian profesional dokter
> - ❌ Tidak boleh dijadikan satu-satunya dasar pengambilan keputusan medis
> - ✅ Harus diverifikasi oleh dokter spesialis radiologi atau neurologi
> - ✅ Sebaiknya digunakan sebagai second opinion atau screening tool
>
> Pasien wajib berkonsultasi dengan tenaga medis bersertifikat untuk diagnosis definitif dan rencana perawatan yang tepat.

## 👨‍💻 Developer

**Rahmat Ardiansyah**
- 🎓 Mahasiswa Teknik Informatika, Universitas Muhammadiyah Riau (UMRI)
- 📍 Pekanbaru, Riau
- 💼 Focus: AI, Computer Vision, Health-Tech
- 🎯 Angkatan: 2022

### Contact
- **University**: Universitas Muhammadiyah Riau
- **Program**: Teknik Informatika
- **Specialization**: Deep Learning & Computer Vision

## 📝 License & Academic Use

Aplikasi ini dikembangkan untuk keperluan akademis (UAS - Ujian Akhir Semester). Diperbolehkan untuk:
- ✅ Penggunaan akademis dan penelitian
- ✅ Pembelajaran dan edukasi
- ✅ Demonstrasi dan presentasi

Tidak diperbolehkan untuk:
- ❌ Penggunaan komersial tanpa izin
- ❌ Diagnosis medis resmi tanpa verifikasi profesional
- ❌ Redistribusi model tanpa atribusi

## 🔧 Troubleshooting

### Problem: Model tidak ditemukan
**Solution**: Pastikan file `VGG16_medium.h5` ada di folder yang sama dengan `app.py`

### Problem: Error protobuf
**Solution**: 
```bash
pip install 'protobuf<5,>=3.20'
```

### Problem: Out of memory
**Solution**: 
- Restart aplikasi
- Gunakan image dengan resolusi lebih kecil
- Pastikan minimal 4GB RAM tersedia

### Problem: Streamlit tidak bisa diakses
**Solution**:
```bash
# Check if running
sudo supervisorctl status streamlit

# Restart service
sudo supervisorctl restart streamlit

# Check logs
tail -f /var/log/supervisor/streamlit.out.log
```

## 🎯 Roadmap & Future Improvements

- [ ] Support untuk format DICOM
- [ ] Batch processing untuk multiple images
- [ ] Integration dengan PACS system
- [ ] Model ensemble untuk akurasi lebih tinggi
- [ ] Real-time video analysis
- [ ] Multi-language support (ID, EN)
- [ ] Export hasil ke format HL7 FHIR
- [ ] Integration dengan Electronic Health Record (EHR)

## 📚 References & Acknowledgments

- VGG16 Architecture: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- TensorFlow Documentation: https://www.tensorflow.org/
- Streamlit Documentation: https://docs.streamlit.io/

## 🙏 Acknowledgments

Terima kasih kepada:
- Dosen pembimbing dan penguji UAS
- Universitas Muhammadiyah Riau (UMRI)
- TensorFlow & Streamlit Community
- Semua pihak yang telah mendukung pengembangan aplikasi ini

---

<div align="center">

**NeuroScan AI v1.0.3**

*Clinical Decision Support System*

© 2026 Rahmat Ardiansyah | Universitas Muhammadiyah Riau

**Developed with ❤️ for Healthcare Innovation**

</div>
