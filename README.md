# Laporan Proyek Machine Learning - Salsabila Ramadhan

## Domain Proyek

Akses terhadap air minum yang aman sangat penting bagi kesehatan, karena air merupakan unsur vital yang dibutuhkan oleh tubuh untuk menjaga fungsi organ-organ, mengatur suhu tubuh, serta mendukung proses metabolisme. Ketersediaan air bersih dan aman juga merupakan faktor kunci dalam pencegahan penyebaran penyakit-penyakit yang terkait dengan air, seperti infeksi saluran pencernaan dan penyakit menular lainnya. Seperti yang dilansir dari [Kemendikbud](https://lldikti5.kemdikbud.go.id/home/detailpost/bahaya-konsumsi-air-yang-tidak-bersih) bahwa jika air yang tidak layak konsumsi akan membuat kondisi tubuh manusia rentan terserang penyakit, salah satunya diare. Oleh karena itu dibutuhkan upaya untuk mengatasi tantangan tersebut. Disinilah peran proyek yang saya buat berfungsi, dengan fokus pada pengembangan model machine learning untuk memprediksi apakah kualitas air. Proyek ini sangat berguna bagi masyarakat. Diantaranya :

- Dengan memastikan akses terhadap air minum yang aman, proyek ini dapat secara langsung meningkatkan kesehatan masyarakat, mengurangi risiko penyakit terkait air.
- Model dapat diterapkan secara berskala besar untuk memantau kualitas air di berbagai daerah, memberikan pemahaman yang lebih baik tentang kondisi air di suatu wilayah.

Model ini akan menggunakan dataset besar yang mencakup berbagai parameter kualitas air, seperti ph, hardness, solids, chloramines dll. Model machine learning akan dilatih dengan menggunakan berbagai macam algoritma machine learning. Seperti KNN, random forest dan naive bayes. Hasil akhir dari model ini adalah model yang dapat memprediksi apakah air layak diminum atau tidak berdasarkan faktor-faktor tertentu.

## Business Understanding

Proyek ini bertujuan untuk mengetahui faktor faktor yang mempengaruhi kualitas air. Faktor-faktor tersebut juga menentukan apakah air dapat dikonsumsi atau tidak oleh manusia. Sehingga, pemahaman mendalam terhadap faktor-faktor yang mempengaruhi kualitas air menjadi krusial dalam menentukan apakah air tersebut aman untuk dikonsumsi oleh manusia. Keberhasilan proyek ini dapat menciptakan dampak positif yang signifikan, baik dari segi kesehatan masyarakat, ekonomi, bisnis, maupun lingkungan. Beberapa dampak yang dapat diantisipasi meliputi:
1. Kesehatan Masyarakat. Dengan adanya prediksi kualitas air yang lebih akurat, masyarakat dapat menghindari konsumsi air yang terkontaminasi, mengurangi risiko penyakit diare, dan meningkatkan kesehatan secara keseluruhan.

2. Ekonomi dan Bisnis. Otoritas pengelola air dapat merencanakan dan mengelola sumber daya air dengan lebih efisien, mengurangi biaya pemeliharaan dan pemulihan akibat pencemaran.

3. Lingkungan. Dengan meminimalkan risiko kontaminasi air, proyek ini dapat membantu melestarikan ekosistem air, memelihara keanekaragaman hayati, dan mengurangi dampak negatif terhadap lingkungan.

Ukuran yang akan diterapkan pada proyek ini untuk memutuskan apakah air layak diminum atau tidak dapat mencakup analisis beberapa parameter yang telah disebutkan. Standar kualitas air pada proyek ini akan berdasarkan standar untuk air minum umumnya yang ditetapkan oleh lembaga kesehatan, seperti Badan Kesehatan Dunia (WHO).

### Problem Statements

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap keamanan air untuk dikonsumsi?
- Apakah semua fitur mempengaruhi tingkat kelayakan konsumsi air?

### Goals

- Mengetahui fitur yang paling berkorelasi dengan kelayakan konsumsi air.
- Membuat model machine learning yang dapat memprediksi apakah air layak diminum atau tidak berdasarkan fitur-fitur yang ada.

## Data Understanding
Data yang digunakan untuk proyek ini diambil dari [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). Data ini memiliki 3277 sample dan merupakan data numerik. Untuk korelasi dari setiap variabel dapat dilihat dari gambar dibawah ini.
![korelasi antar variabel](https://github.com/salsabilar311/Predictive-Analytics_Water-Quality/assets/98375087/1923d7f6-ed65-43fe-a604-5ab3e20e397a)

Dapat dilihat dari gambar diatas bahwa conductivity dan organic_carbon memiliki relasi yang lemah terhadap potability. Sedangkan parameter lainnya memiliki relasi yang relatif sama terhadap potability. Hal ini menunjukkan bahwa, meskipun kedua parameter tersebut dapat memberikan informasi tentang sifat fisik dan kimia air, kontribusinya terhadap kelayakan air minum mungkin tidak begitu dominan dibandingkan dengan parameter lain seperti pH, hardness, dan kloramin. Oleh karena itu, dalam pengembangan model machine learning untuk memprediksi kelayakan minum air, perlu mempertimbangkan dengan cermat bobot atau signifikansi relatif dari masing-masing parameter. Selain itu, analisis ini dapat memberikan dorongan untuk lebih mendalam memahami faktor-faktor yang memengaruhi kelayakan air minum. Analisis antar variabel:

- Variabel pH memiliki relasi kuat dengan variabel hardness dengan tingkat korelasi sebesar 0.14. Angka yang cukup positif, ini menunjukkan bahwa tingkat keasaman air cenderung meningkat seiring dengan peningkatan tingkat kekerasan air.
- Variabel solids dan sulfate memiliki relasi yang lemah. Yaitu -0.14, nilai ini berbanding terbalik dengan relasi pH dan hardness. Ini menunjukkan bahwa ketika nilai solids meningkat, nilai sulfate cenderung menurun, dan sebaliknya. Ini dapat diartikan sebagai adanya hubungan invers antara kandungan padatan terlarut (solids) dan kandungan sulfate dalam air.

### Variabel-variabel pada Water Quality dataset adalah sebagai berikut:
- pH: pH air 1. (0 hingga 14). pH berguna untuk mengukur tingkat keasaman atau kebasaan air. Pengukuran pH yaitu nilai 7 adalah netral (air murni), kurang dari 7 bersifat asam, dan lebih dari 7 bersifat basa.
- Hardness: Kapasitas air untuk mengendapkan sabun dalam mg/L. Tingkat kekerasan air dapat memengaruhi efektivitas sabun dan deterjen.
- Solids: Total padatan terlarut dalam ppm. Tingkat padatan terlarut dapat memengaruhi rasa, warna, dan kejernihan air.
- Chloramines: Jumlah Kloramin dalam ppm. Kloramin digunakan untuk menjaga kualitas air dengan membunuh bakteri dan mikroorganisme lainnya.
- Sulfate: Jumlah Sulfat yang dilarutkan dalam mg/L. Tingkat sulfat dapat mempengaruhi rasa air dan menyebabkan masalah kesehatan jika melebihi batas tertentu.
- Conductivity: Konduktivitas listrik air dalam μS/cm. Konduktivitas dapat mencerminkan kandungan ion dalam air.
- Organic_carbon: Jumlah karbon organik dalam ppm. Karbon organik dapat berasal dari materi organik yang terurai dalam air.
- Trihalomethanes: Jumlah Trihalometana dalam μg/L. Trihalometana dapat memiliki dampak kesehatan jika melebihi batas tertentu.
- Turbidity: Ukuran properti pemancar cahaya air dalam NTU. Kekeruhan dapat mempengaruhi penampilan dan kejernihan air.
- Potability: Menunjukkan apakah air aman untuk dikonsumsi manusia. Dapat diminum -1 dan Tidak dapat diminum -0

## Data Preparation
- Membuang missing value

  Pada tahap pre-processing data, akan dilakukan penghapusan terhadup nilai yang nulll. Hal ini dilakukan agar tidak mempengaruhi kualitas model pada saat melakukan prediksi.
- Menghapus outlier

   Pada tahap ini, menganalisis distribusi data dan mengidentifikasi serta menghapus outlier dari variabel-variabel numeric. Hal ini dilakukan agar model dapat meningkatkan akurasi. Teknik yang dipakai untuk menghapus outlier adalah dengan menggunakan IQR. Pada dasarnya teknik ini akan menghapus nilai yang berada di luar ambang batas. 
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.

  Pada tahap ini dataset akan menjadi dua bagian yaitu train set (untuk melatih model) dan test set (untuk menguji model). Ini membantu menilai sejauh mana model dapat umum digunakan pada data baru.
- Normalisasi menggunakan StandardScaler

  Pada langkah ini, data akan dinormalisasi menggunakan StandardScaler agar setiap variabel memiliki skala yang serupa. Hal ini membantu untuk model melakukan prediksi. Langkah-langkah untuk melakukan normalisasi menggunakan standard scaler diantaranya yaitu:

    1. Import library
    2. Inisialisasi StandardScaler
    3. Fit data pelatihan
    4. Transform data pelatihan

## Modeling
Pada proses modeling ini saya menggunakan 3 algoritma yaitu :
1. KNN

    KNN adalah metode klasifikasi berbasis instan yang bekerja dengan cara menemukan kelas mayoritas dari k tetangga terdekat suatu titik data yang belum diketahui kelasnya. KNN dipilih karena sifatnya yang non-parametrik dan mampu menangani pola kompleks dalam data tanpa membuat asumsi tertentu tentang distribusi data. Cocok untuk kasus ini karena dataset memiliki berbagai parameter kualitas air yang kompleks dan tidak memiliki asumsi tertentu.

2. Random forest

    Random Forest adalah metode ensemble yang menggabungkan beberapa model pohon keputusan untuk meningkatkan kinerja dan kestabilan prediksi. Random Forest dipilih karena sifatnya sebagai metode ensemble yang menggabungkan beberapa pohon keputusan. Mampu menangani data yang kompleks dan cenderung overfitting, sehingga dapat memberikan hasil yang lebih stabil dan akurat. Ini menjadikan model dapat memprediksi dataset dengan banyak parameter dan meningkatkan akurasi dari model. Cocok untuk kasus ini karena dataset dari proyek ini memiliki 10 parameter.

3. Naive bayes

    Naive Bayes adalah metode klasifikasi berbasis probabilitas yang menggunakan teorema Bayes dengan asumsi independensi antar fitur. Naive Bayes dipilih karena kesederhanaannya dan ketangguhannya dalam menangani data dengan dimensi tinggi serta dapat memberikan hasil yang baik bahkan dengan asumsi independensi yang naif. Cocok dengan dataset yang memiliki sejumlah variabel penentu seperti proyek ini.

## Evaluation
Metrik evaluasi yang digunakan adalah akurasi. Karena ini merupakan masalah klasifikasi jadi saya menggunakan akurasi sebagai metrik evaluasi. Berdasarkan proses modeling didapat hasil:
| KNN | Random Forest | Naive Bayes |
|:--------------:|:--------------:|:--------------:|
| 0.57 | 0.58    | 0.41   |

Dapat dilihat dari tabel diatas bahwa algoritma random forest memiliki skor akurasi yang terbesar yaitu **0.58**. Skor akurasi Random Forest 0.58, artinya 58% dari total prediksi yang dilakukan oleh model Random Forest benar. Oleh karena itu random forest adalah algoritma yang akan digunakan untuk melakukan prediksi selanjutnya. Namun, untuk pengembangan proyek lebih lanjut, diperlukan evaluasi yang lebih mendalam terhadap kinerja model random forest ini. Meskipun skor akurasi dapat memberikan gambaran umum tentang seberapa baik model melakukan prediksi, beberapa metrik evaluasi tambahan perlu dipertimbangkan. Seperti precision, recall, dan F1-score. Dan perlu diselidiki juga mengapa performa Naive Bayes kurang  memuaskan. Apakah asumsi independensi dalam Naive Bayes sesuai dengan karakteristik data proyek ini. Untuk mengetahui hal itu diperlukan peninjauan terhadap confusion matrix yang dapat memberikan informasi lebih rinci tentang performa model, terutama dalam hal false positives dan false negatives.
