# Penghilangan Kabut pada Citra Menggunakan Dark Channel Prior
## Implementasi From Scratch dengan Perbandingan Baseline CLAHE

---

**Mata Kuliah:** Visi Komputer  
**Jenis Tugas:** Final Project Semester 2025/2026  
**Nama:** Ikrar Gempur Tirani  
**NIM:** D121231015  

---

## Daftar Isi

1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Latar Belakang Teoritis](#latar-belakang-teoritis)
3. [Fitur dan Komponen Implementasi](#fitur-dan-komponen-implementasi)
4. [Struktur Proyek](#struktur-proyek)
5. [Dependensi dan Persyaratan](#dependensi-dan-persyaratan)
6. [Instalasi dan Konfigurasi](#instalasi-dan-konfigurasi)
7. [Data Masukan](#data-masukan)
8. [Cara Penggunaan](#cara-penggunaan)
9. [Deskripsi Algoritma](#deskripsi-algoritma)
10. [Metrik Evaluasi](#metrik-evaluasi)
11. [Hasil dan Analisis](#hasil-dan-analisis)
12. [Keterbatasan Metode](#keterbatasan-metode)
13. [Referensi](#referensi)

---

## Deskripsi Proyek

Proyek ini mengimplementasikan algoritma **Dark Channel Prior (DCP)** untuk penghilangan kabut (*image dehazing*) pada citra digital, seluruhnya ditulis **dari awal** (*from scratch*) menggunakan pustaka NumPy tanpa ketergantungan pada OpenCV, SciPy, maupun Scikit-image. Implementasi dibandingkan secara kuantitatif terhadap metode baseline **Contrast Limited Adaptive Histogram Equalization (CLAHE)** yang juga diimplementasikan secara mandiri.

Validasi dilakukan pada dua skenario: (1) citra sintetis berkabut dengan *ground truth* yang diketahui, memungkinkan penghitungan metrik PSNR dan SSIM yang akurat; dan (2) citra berkabut nyata (*real-world*) menggunakan metrik tanpa-acuan (*no-reference*) berupa gradien rata-rata sebagai indikator ketajaman.

---

## Latar Belakang Teoritis

### Model Hamburan Atmosfer

Pembentukan citra berkabut dimodelkan oleh persamaan hamburan atmosfer berikut:

```
I(x) = J(x) * t(x) + A * (1 - t(x))
```

Keterangan:
- `I(x)` : citra yang teramati (berkabut)
- `J(x)` : citra laten yang ingin dipulihkan (jernih)
- `t(x)` : peta transmisi (transmission map), menggambarkan porsi cahaya objek yang mencapai kamera
- `A`    : intensitas cahaya atmosfer (*atmospheric light*)

Proses dehazing pada intinya adalah membalikkan persamaan di atas, yaitu mengestimasi `t(x)` dan `A` sehingga `J(x)` dapat dihitung.

### Dark Channel Prior

**Dark Channel Prior** adalah observasi empiris bahwa pada citra luar ruangan (*outdoor*) yang tidak berkabut, setiap patch lokal hampir selalu memiliki minimal satu piksel dengan intensitas sangat rendah di salah satu kanal warnanya:

```
J_dark(x) = min_{c in {R,G,B}} ( min_{y in patch(x)} J^c(y) ) ≈ 0
```

Prinsip ini diperoleh karena pada citra nyata terdapat bayangan, permukaan berwarna gelap, atau objek berwarna pekat yang selalu hadir di setiap area gambar. Kabut melanggar prior ini secara sistematis karena kabut menaikkan nilai minimum lokal mendekati nilai cahaya atmosfer `A`. Dengan memanfaatkan pelanggaran inilah transmisi dapat diestimasi.

---

## Fitur dan Komponen Implementasi

Seluruh komponen berikut diimplementasikan dari awal menggunakan **NumPy saja**:

| Komponen | Deskripsi |
|---|---|
| `rgb2gray` | Konversi luminansi menggunakan bobot standar Rec.709 |
| `resize_bilinear` | Pengubahan ukuran citra dengan interpolasi bilinear dua-sumbu |
| `min_filter_2d` | Minimum filter 2D (operasi erosi morfologi) dengan pendekatan separable satu dimensi |
| `box_mean` | Filter rata-rata lokal menggunakan *summed-area table* (integral image) berkompleksitas linear |
| `dark_channel` | Komputasi dark channel: minimum antar-kanal dilanjutkan minimum spasial lokal |
| `atmospheric_light` | Estimasi cahaya atmosfer: seleksi 0,1% piksel terterang pada dark channel |
| `estimate_transmission` | Estimasi transmission map kasar berdasarkan Dark Channel Prior |
| `guided_filter` | Penyempurnaan transmission menggunakan filter linier lokal (*guided filter*) |
| `recover` | Pemulihan citra jernih dari model hamburan atmosfer |
| `dehaze_dcp` | Pipeline lengkap Dark Channel Prior end-to-end |
| `clahe` | CLAHE pada ruang warna YCbCr dengan interpolasi bilinear antartile |
| `psnr` | Metrik Peak Signal-to-Noise Ratio |
| `ssim` | Metrik Structural Similarity Index Measure (SSIM) per kanal |
| `avg_gradient` | Metrik gradien rata-rata sebagai ukuran ketajaman tanpa-acuan |
| `synthesize_haze` | Sintesis kabut sintetis menggunakan model atmosfer (untuk ground truth) |

---

## Struktur Proyek

```
.
├── dehazing_dcp.ipynb      # Notebook utama berisi seluruh implementasi dan analisis
├── gt_clear.jpg            # Citra jernih (ground truth) untuk sintesis dan evaluasi
├── nyata_kota.jpg          # Citra berkabut nyata: skenario perkotaan (keberhasilan)
├── nyata_padang.jpg        # Citra berkabut nyata: skenario padang (keterbatasan)
└── README.md               # Dokumentasi proyek ini
```

---

## Dependensi dan Persyaratan

### Peran Setiap Pustaka

| Pustaka | Versi Minimum | Peran dalam Proyek |
|---|---|---|
| Python | 3.8+ | Bahasa pemrograman |
| NumPy | 1.21+ | Seluruh komputasi algoritma |
| Matplotlib | 3.4+ | Visualisasi hasil semata (bukan algoritma) |
| Pillow (PIL) | 8.0+ | Pembacaan berkas gambar saja (I/O, bukan algoritma) |
| Jupyter | 6.0+ | Lingkungan eksekusi notebook |

> Catatan penting: OpenCV, SciPy, dan Scikit-image secara eksplisit tidak digunakan. Seluruh operasi pemrosesan citra — termasuk erosi, box filter, guided filter, konversi warna, CLAHE, dan SSIM — diimplementasikan secara mandiri di atas NumPy.

---

## Instalasi dan Konfigurasi

### 1. Clone atau Unduh Repositori

```bash
git clone https://github.com/Ikrar06/dehazing-dcp-from-scratch
cd final
```

### 2. Buat Lingkungan Virtual (Opsional tapi Disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instal Dependensi

```bash
pip install numpy matplotlib Pillow jupyter
```

Atau menggunakan file requirements:

```bash
pip install -r requirements.txt
```

Contoh isi `requirements.txt`:

```
numpy>=1.21
matplotlib>=3.4
Pillow>=8.0
jupyter>=6.0
```

### 4. Jalankan Notebook

```bash
jupyter notebook dehazing_dcp.ipynb
```

---

## Data Masukan

Tiga berkas citra diperlukan sebelum menjalankan notebook:

| Berkas | Deskripsi | Keterangan |
|---|---|---|
| `gt_clear.jpg` | Citra jernih berkualitas tinggi, idealnya pemandangan perkotaan atau alam terbuka | Digunakan sebagai ground truth; kabut sintetis ditambahkan secara komputasional |
| `nyata_kota.jpg` | Foto kabut nyata kondisi perkotaan | Digunakan pada Sel 11 sebagai skenario keberhasilan |
| `nyata_padang.jpg` | Foto kabut nyata kondisi terbuka atau berawan | Digunakan pada Sel 11 sebagai skenario keterbatasan |

Rekomendasi ukuran: lebar antara 600--1200 piksel. Citra yang lebih besar akan diresize otomatis oleh fungsi `load_image` ke lebar maksimum yang ditentukan (default 800--900 piksel).

---

## Cara Penggunaan

Notebook disusun secara sekuensial dan dapat dijalankan langsung dari awal hingga akhir dengan memilih **Run All** (`Kernel > Restart & Run All`). Berikut ringkasan setiap seksi:

| Sel | Judul | Isi |
|---|---|---|
| 0 | Setup | Import pustaka dan konfigurasi Matplotlib |
| 1 | Teori Singkat | Penjelasan model hamburan atmosfer dan Dark Channel Prior |
| 2 | Fungsi Dasar | Implementasi from scratch: rgb2gray, bilinear resize, min-filter, box-filter |
| 3 | Load Citra dan Sintesis Kabut | Memuat ground truth dan mensintesis kabut sintetis |
| 4 | Dark Channel Prior | Implementasi penuh pipeline DCP |
| 5 | Baseline CLAHE | Implementasi CLAHE pada ruang warna YCbCr |
| 6 | Metrik | Implementasi PSNR, SSIM, dan gradien rata-rata |
| 7 | Jalankan dan Bandingkan | Eksekusi DCP dan CLAHE, visualisasi berdampingan |
| 8 | Visualisasi Tahap Antara | Dark channel, transmission kasar, transmission terhaluskan |
| 9 | Evaluasi Kuantitatif | Tabel dan grafik batang PSNR dan SSIM ketiga metode |
| 10 | Variasi Parameter | Kurva SSIM terhadap intensitas kabut (beta) dan ukuran patch |
| 11 | Uji Citra Nyata | Dehazing pada dua foto kabut nyata dengan gradien rata-rata |

---

## Deskripsi Algoritma

### A. Pipeline Dark Channel Prior

Tahapan algoritma DCP secara lengkap:

**Langkah 1 — Komputasi Dark Channel**

Minimum spasial lokal (erosi) diterapkan pada citra minimum antar-kanal:

```
dark(x) = min_{c} ( min_{y in patch(x)} I^c(y) )
```

Operasi ini setara dengan erosi morfologi menggunakan elemen struktural kotak berukuran `patch x patch`. Dalam implementasi ini dikerjakan secara separable: dua kali penyapuan satu dimensi sehingga kompleksitas tetap linear terhadap jumlah piksel.

**Langkah 2 — Estimasi Cahaya Atmosfer**

Kandidat piksel diambil dari 0,1% piksel tertinggi pada dark channel. Di antara kandidat tersebut, piksel dengan intensitas total (R+G+B) tertinggi ditetapkan sebagai estimasi `A`.

**Langkah 3 — Estimasi Transmission Map Kasar**

Dengan menggunakan dark channel prior pada citra yang telah dinormalisasi terhadap `A`:

```
t_kasar(x) = 1 - omega * dark_channel(I / A)
```

Parameter `omega = 0.95` mempertahankan sedikit kabut pada objek jauh agar tampilan tetap realistis.

**Langkah 4 — Penyempurnaan Transmission dengan Guided Filter**

Transmission map kasar memiliki artefak blok akibat operasi patch. Guided filter dengan panduan citra grayscale memperhalus map sambil mempertahankan tepian objek:

```
t_halus(x) = guided_filter(grayscale(I), t_kasar, r=40, eps=1e-3)
```

Guided filter diimplementasikan menggunakan box_mean (summed-area table) sehingga kompleksitasnya tetap linear.

**Langkah 5 — Pemulihan Citra**

Model hamburan dibalikkan dengan penjepitan transmission minimum `t0 = 0.1`:

```
J(x) = (I(x) - A) / max(t(x), t0) + A
```

### B. Pipeline CLAHE (Baseline)

1. Konversi RGB ke YCbCr; proses hanya pada kanal luminansi Y
2. Bagi citra menjadi grid tile (default 8x8)
3. Untuk setiap tile: hitung histogram, potong pada batas clip, distribusikan kelebihan secara merata, normalisasi CDF menjadi LUT
4. Rekonstruksi piksel dengan interpolasi bilinear antara empat LUT tile terdekat
5. Gabungkan kembali Y hasil CLAHE dengan Cb dan Cr asli, konversi ke RGB

---

## Metrik Evaluasi

### Metrik Referensi Penuh (Full-Reference)

Digunakan pada eksperimen sintetis yang memiliki ground truth `J_clear`:

**PSNR (Peak Signal-to-Noise Ratio)**

```
PSNR = 10 * log10( 1 / MSE )
MSE  = mean( (J_pred - J_clear)^2 )
```

Satuan dalam desibel (dB). Nilai lebih tinggi menunjukkan hasil lebih baik.

**SSIM (Structural Similarity Index Measure)**

Dihitung per kanal menggunakan jendela seragam 7x7 dengan koreksi bias kovarians sampel. SSIM akhir adalah rata-rata dari tiga kanal warna. Nilainya berkisar antara -1 hingga 1; nilai mendekati 1 berarti lebih baik.

### Metrik Tanpa-Acuan (No-Reference)

Digunakan pada uji citra nyata yang tidak memiliki ground truth:

**Gradien Rata-Rata**

```
avg_gradient = mean( sqrt( (dI/dx)^2 + (dI/dy)^2 ) )
```

Dihitung pada citra grayscale menggunakan `numpy.gradient`. Nilai yang lebih tinggi mengindikasikan citra lebih tajam, namun metrik ini hanya indikatif karena tidak membedakan ketajaman nyata dari artefak.

---

## Hasil dan Analisis

### Eksperimen Sintetis

Pada pengujian dengan citra sintetis (beta=1.6, A=0.85), DCP menunjukkan perbaikan signifikan dibandingkan CLAHE pada kedua metrik:

| Metode | PSNR (dB) | SSIM |
|---|---|---|
| Berkabut (tanpa proses) | (baseline) | (baseline) |
| CLAHE (baseline) | meningkat moderat | meningkat moderat |
| DCP (metode utama) | meningkat tajam | meningkat tajam |

Keunggulan DCP atas CLAHE bersumber dari perbedaan pendekatan mendasar: CLAHE hanya memanipulasi kontras histogram secara lokal tanpa model fisika, sementara DCP secara eksplisit memodelkan dan membalikkan proses pembentukan kabut sesuai fisika hamburan atmosfer.

### Eksperimen Variasi Parameter

- **Intensitas kabut (beta):** SSIM DCP menurun seiring beta meningkat, namun secara konsisten lebih tinggi dari CLAHE di seluruh rentang beta yang diuji (0.8 hingga 2.5).
- **Ukuran patch dark channel:** terdapat nilai optimal di sekitar patch=15; patch terlalu kecil menghasilkan estimasi noise, patch terlalu besar mengakibatkan halo di sekitar objek.

### Uji Citra Nyata

| Skenario | Gradien Sebelum | Gradien Sesudah | Keterangan |
|---|---|---|---|
| Kota (keberhasilan) | rendah | meningkat | Detail pulih, artefak minimal |
| Padang (keterbatasan) | moderat | sedikit turun | Area langit memicu artefak |

---

## Keterbatasan Metode

1. **Artefak pada area langit dan objek putih besar**: Dark Channel Prior mengasumsikan tiap patch memiliki piksel gelap. Area langit cerah dan objek berwarna putih melanggar asumsi ini sehingga transmission diestimasi terlalu rendah, mengakibatkan over-enhancement dan munculnya artefak warna.

2. **Ketergantungan pada parameter manual**: Nilai `patch`, `omega`, `t0`, `r` (radius guided filter), dan `eps` memengaruhi kualitas hasil secara signifikan dan perlu disesuaikan secara manual tergantung karakteristik input.

3. **Kompleksitas komputasi guided filter**: Meskipun box_mean berbasis integral image memiliki kompleksitas O(N), radius filter yang besar (r=40) tetap memerlukan alokasi memori besar untuk array padding dan cumsum.

4. **Keterbatasan metrik no-reference**: Gradien rata-rata hanya indikatif; artefak intensitas tinggi dapat meningkatkan nilai gradien tanpa benar-benar meningkatkan kualitas visual.

5. **Asumsi kedalaman sederhana dalam sintesis**: Model sintesis menggunakan kedalaman linier searah vertikal, yang merupakan penyederhanaan dari kondisi kabut nyata yang memiliki distribusi tiga dimensi.

---

## Referensi

1. He, K., Sun, J., & Tang, X. (2011). Single image haze removal using dark channel prior. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(12), 2341--2353. https://doi.org/10.1109/TPAMI.2010.168

2. He, K., Sun, J., & Tang, X. (2013). Guided image filtering. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(6), 1397--1409. https://doi.org/10.1109/TPAMI.2012.213

3. ITU-R. (2011). *BT.601-7: Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios*. International Telecommunication Union.

4. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600--612. https://doi.org/10.1109/TIP.2003.819861

5. Pizer, S. M., Amburn, E. P., Austin, J. D., Cromartie, R., Geselowitz, A., Greer, T., ter Haar Romeny, B., Zimmerman, J. B., & Zuiderveld, K. (1987). *Adaptive histogram equalization and its variations. Computer Vision, Graphics, and Image Processing, 39(3)*, 355–368. 

6. Crow, F. C. (1984). *Summed-area tables for texture mapping. ACM SIGGRAPH Computer Graphics, 18(3)*, 207–212. 

7. Li, B., Ren, W., Fu, D., Tao, D., Feng, D., Zeng, W., & Wang, Z. (2018). *Benchmarking single-image dehazing and beyond. IEEE Transactions on Image Processing, 28(1)*, 492–505.

8. Narasimhan, S. G., & Nayar, S. K. (2002). *Vision and the atmosphere. International Journal of Computer Vision, 48(3)*, 233–254. 

---

*Final Project — Mata Kuliah Visi Komputer, 2025/2026*  
*Ikrar Gempur Tirani — D121231015*
