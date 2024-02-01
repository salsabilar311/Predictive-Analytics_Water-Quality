# Laporan Proyek Machine Learning - Salsabila Ramadhan

## Domain Proyek

Kualitas air adalah salah satu problem yang sering terjadi saat ini. Kualitas air dipengaruhi oleh beberapa faktor yaitu pencemaran udara, aktivitas pertanian, pencemaran industri dll. Oleh karena itu dibutuhkan upaya serius dan terkoordinasi untuk mengatasi tantangan kualitas air ini. Disinilah peran proyek yang saya buat berfungsi untuk memprediksi kualitas air dan menggabungkan kecerdasan buatan serta analisis data untuk memberikan solusi inovatif. Dengan memanfaatkan model prediktif, proyek ini dapat memberikan pemahaman mendalam tentang faktor-faktor yang memengaruhi kualitas air.

## Business Understanding

Proyek ini bertujuan untuk mengetahui faktor faktor yang mempengaruhi kualitas air. Faktor-faktor tersebut juga menentukan apakah air dapat dikonsumsi atau tidak oleh manusia. Sehingga, pemahaman mendalam terhadap faktor-faktor yang mempengaruhi kualitas air menjadi krusial dalam menentukan apakah air tersebut aman untuk dikonsumsi oleh manusia.

### Problem Statements

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap keamanan air untuk dikonsumsi?
- Apakah semua fitur mempengaruhi tingkat kelayakan konsumsi air?

### Goals

- Mengetahui fitur yang paling berkorelasi dengan kelayakan konsumsi air.
- Membuat model machine learning yang dapat memprediksi apakah air layak diminum atau tidak berdasarkan fitur-fitur yang ada.

## Data Understanding
Data yang digunakan untuk proyek ini diambil dari [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). Data ini memiliki 3277 sample dan merupakan data numerik.

### Variabel-variabel pada Water Quality dataset adalah sebagai berikut:
- pH: pH air 1. (0 hingga 14).
- Hardness: Kapasitas air untuk mengendapkan sabun dalam mg/L.
- Solids: Total padatan terlarut dalam ppm.
- Chloramines: Jumlah Kloramin dalam ppm.
- Sulfate: Jumlah Sulfat yang dilarutkan dalam mg/L.
- Conductivity: Konduktivitas listrik air dalam μS/cm.
- Organic_carbon: Jumlah karbon organik dalam ppm.
- Trihalomethanes: Jumlah Trihalometana dalam μg/L.
- Turbidity: Ukuran properti pemancar cahaya air dalam NTU.
- Potability: Menunjukkan apakah air aman untuk dikonsumsi manusia. Dapat diminum -1 dan Tidak dapat diminum -0

## Data Preparation
- Membuang missing value
- Menghapus outlier
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.
- Normalisasi menggunakan StandardScaler

## Modeling
Pada proses modeling ini saya menggunakan 3 algoritma yaitu :
1. KKN
2. Random forest
3. Naive bayes

## Evaluation
Metrik evaluasi yang digunakan adalah accuracy. Karena ini merupakan masalah klasifikasi jadi saya menggunakan accuracy sebagai metrik evaluasi. Berdasarkan proses modeling didapat hasil:
| KNN | Random Forest | Naive Bayes |
|:--------------:|:--------------:|:--------------:|
| 0.57 | 0.58    | 0.41   |

Dapat dilihat dari tabel diatas bahwa algoritma random forest memiliki skor accuracy yang terbesar yaitu **0.58**. Oleh karena itu random forest adalah algoritma yang akan digunakan untuk melakukan prediksi selanjutnya. Namun, untuk pengembangan proyek lebih lanjut, diperlukan evaluasi yang lebih mendalam terhadap kinerja model random forest ini. Meskipun skor akurasi dapat memberikan gambaran umum tentang seberapa baik model melakukan prediksi, beberapa metrik evaluasi tambahan perlu dipertimbangkan. 
