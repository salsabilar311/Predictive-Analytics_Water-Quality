# Laporan Proyek Machine Learning - Salsabila Ramadhan

## Domain Proyek

Akses terhadap air minum yang aman sangat penting bagi kesehatan, karena air merupakan unsur vital yang dibutuhkan oleh tubuh untuk menjaga fungsi organ-organ, mengatur suhu tubuh, serta mendukung proses metabolisme. Ketersediaan air bersih dan aman juga merupakan faktor kunci dalam pencegahan penyebaran penyakit-penyakit yang terkait dengan air, seperti infeksi saluran pencernaan dan penyakit menular lainnya. Seperti yang dilansir dari [Kemendikbud](https://lldikti5.kemdikbud.go.id/home/detailpost/bahaya-konsumsi-air-yang-tidak-bersih) bahwa jika air yang tidak layak konsumsi akan membuat kondisi tubuh manusia rentan terserang penyakit, salah satunya diare [1]. Oleh karena itu dibutuhkan upaya untuk mengatasi tantangan tersebut. 
Disinilah peran proyek yang ini berfungsi, dengan fokus pada pengembangan model machine learning untuk memprediksi apakah kualitas air. Proyek ini sangat berguna bagi masyarakat. Diantaranya :

- Dengan memastikan akses terhadap air minum yang aman, proyek ini dapat secara langsung meningkatkan kesehatan masyarakat, mengurangi risiko penyakit terkait air.
- Model dapat diterapkan secara berskala besar untuk memantau kualitas air di berbagai daerah, memberikan pemahaman yang lebih baik tentang kondisi air di suatu wilayah.

Model ini akan menggunakan dataset besar yang mencakup berbagai parameter kualitas air, seperti ph, hardness, solids, chloramines dll. Model machine learning akan dilatih dengan menggunakan berbagai macam algoritma machine learning. Seperti KNN, random forest dan naive bayes. Hasil akhir dari model ini adalah model yang dapat memprediksi apakah air layak diminum atau tidak berdasarkan faktor-faktor tertentu. Untuk menerapkannya pada masyarakat, user memerlukan sebuah data kualitas air yang berisi beberapa parameter pengukur. Lalu user dapat memasukkan data tersebut ke model yang sudah dilatih. Model akan melakukan prediksi dan menghasilkan output berupa layak atau tidak layak air untuk diminum. Diharapkan model ini mampu memprediksi kelayakan air minum sesuai dengan parameter yang ada. Prediksi ini nantinya dijadikan acuan bagi masyarakat dalam menentukan apakah air dapat diminum atau tidak.

## Business Understanding

Proyek ini bertujuan untuk mengetahui faktor faktor yang mempengaruhi kualitas air. Faktor-faktor tersebut juga menentukan apakah air dapat dikonsumsi atau tidak oleh manusia. Sehingga, pemahaman mendalam terhadap faktor-faktor yang mempengaruhi kualitas air menjadi krusial dalam menentukan apakah air tersebut aman untuk dikonsumsi oleh manusia. Keberhasilan proyek ini dapat menciptakan dampak positif yang signifikan, baik dari segi kesehatan masyarakat, ekonomi, bisnis, maupun lingkungan. Beberapa dampak dari hasil proyek ini meliputi:
1. Kesehatan Masyarakat. Dengan adanya prediksi kualitas air yang lebih akurat, masyarakat dapat menghindari konsumsi air yang terkontaminasi, mengurangi risiko penyakit diare, dan meningkatkan kesehatan secara keseluruhan. Seperti yang dilansir dalam [expertindo](https://expertindo-training.com/pengaruh-kualitas-air-minum-terhadap-kesehatan-masyarakat/) bahwa
    > Kualitas air minum memiliki hubungan krusial dengan kesehatan manusia, dan air yang terkontaminasi atau mengandung bahan kimia berbahaya dapat membawa risiko serius bagi kesejahteraan masyarakat. Air yang tercemar dapat mengandung berbagai mikroorganisme patogen seperti bakteri, virus, dan parasit. Saat diminum atau digunakan dalam kegiatan sehari-hari seperti mandi dan memasak, mikroorganisme ini dapat masuk ke dalam tubuh manusia dan menyebabkan infeksi saluran pencernaan seperti diare, kolera, dan tifus. Kontaminan kimia seperti logam berat (seperti timbal dan merkuri) atau bahan kimia industri yang mencemari air juga memiliki potensi berbahaya. Paparan berulang terhadap kontaminan semacam ini dapat menyebabkan keracunan, merusak organ vital seperti hati dan ginjal, dan bahkan meningkatkan risiko kanker[2].

    Oleh karena itu proyek ini dapat membantu untuk mencegah hal tersebut terjadi.

2. Ekonomi dan Bisnis. Otoritas pengelola air dapat merencanakan dan mengelola sumber daya air dengan lebih efisien, mengurangi biaya pemeliharaan dan pemulihan akibat pencemaran. Selain pengelolaan air, ternyata proyek ini berpengaruh yang signifikan terhadap ekonomi dan bisnis. Pencemaran air dapat menghambat pertumbuhan ekonomi global dan menurunkan produktivitas. Pabrik-pabrik dan bisnis yang bergantung pada pasokan air bersih dapat terganggu jika pasokan air tercemar atau terbatas. Seperti yang dilansir di [ekonomi.bisnis](https://ekonomi.bisnis.com/read/20190821/9/1139261/waduh-pencemaran-air-ternyata-bisa-hambat-pertumbuhan-ekonomi-global) bahwa “Memburuknya kualitas air menghambat pertumbuhan ekonomi, menurunkan kondisi kesehatan, mengurangi produksi pangan, dan memperburuk kemiskinan di banyak negara,” kata Presiden Bank Dunia David Malpass melalui sebuah pernyataan seperti dikutip Bisnis.com, Rabu (21/8/2019)[3]. Oleh karena itu diharapkan proyek ini dapat membantu meningkatkan pertumbuhan ekonomi.

3. Lingkungan. Dengan meminimalkan risiko kontaminasi air, proyek ini dapat membantu melestarikan ekosistem air, memelihara keanekaragaman hayati, dan mengurangi dampak negatif terhadap lingkungan. Sehingga dengan adanya proyek ini dapat melestarikan lingkungan dengan memelihara keanekaragaman hayati. Karena pada dasarnya air sangat dibutuhkan oleh setiap mahluk hidup. Dengan adanya air bersih maka kualitas lingkungan akan terjaga.

Ukuran yang akan diterapkan pada proyek ini untuk memutuskan apakah air layak diminum atau tidak dapat mencakup analisis beberapa parameter yang telah disebutkan. Standar kualitas air pada proyek ini akan berdasarkan standar untuk air minum umumnya yang ditetapkan oleh lembaga kesehatan, seperti Badan Kesehatan Dunia (WHO).

### Problem Statements

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap keamanan air untuk dikonsumsi?
- Apakah semua fitur mempengaruhi tingkat kelayakan konsumsi air?

### Goals

- Mengetahui fitur yang paling berkorelasi dengan kelayakan konsumsi air.
- Membuat model machine learning yang dapat memprediksi apakah air layak diminum atau tidak berdasarkan fitur-fitur yang ada.

## Data Understanding
Data yang digunakan untuk proyek ini diambil dari [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). Berikut beberapa informasi pada dataset :

- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 3277 sample dengan 10 fitur.
- Dataset memiliki 9 fitur bertipe float64 dan 1 fitur bertipe int64.
- Terdapat banyak missing value dalam dataset.

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

Untuk korelasi dari setiap variabel dapat dilihat dari gambar dibawah ini.
![korelasi antar variabel](https://drive.google.com/file/d/1LMf9q2RqieHbYnqVkv18vOmL5eTlyv4p/view?usp=drive_link)

Dapat dilihat dari gambar diatas bahwa conductivity dan organic_carbon memiliki relasi yang lemah terhadap potability. Sedangkan parameter lainnya memiliki relasi yang relatif sama terhadap potability. Hal ini menunjukkan bahwa, meskipun kedua parameter tersebut dapat memberikan informasi tentang sifat fisik dan kimia air, kontribusinya terhadap kelayakan air minum mungkin tidak begitu dominan dibandingkan dengan parameter lain seperti pH, hardness, dan kloramin. Oleh karena itu, dalam pengembangan model machine learning untuk memprediksi kelayakan minum air, perlu mempertimbangkan dengan cermat bobot atau signifikansi relatif dari masing-masing parameter. Selain itu, analisis ini dapat memberikan dorongan untuk lebih mendalam memahami faktor-faktor yang memengaruhi kelayakan air minum. Analisis antar variabel:

- Variabel pH memiliki relasi kuat dengan variabel hardness dengan tingkat korelasi sebesar 0.14. Angka yang cukup positif, ini menunjukkan bahwa tingkat keasaman air cenderung meningkat seiring dengan peningkatan tingkat kekerasan air.
- Variabel solids dan sulfate memiliki relasi yang lemah. Yaitu -0.14, nilai ini berbanding terbalik dengan relasi pH dan hardness. Ini menunjukkan bahwa ketika nilai solids meningkat, nilai sulfate cenderung menurun, dan sebaliknya. Ini dapat diartikan sebagai adanya hubungan invers antara kandungan padatan terlarut (solids) dan kandungan sulfate dalam air.

## Data Preparation
- Membuang missing value

  Pada tahap pre-processing data, akan dilakukan penghapusan terhadap nilai yang nulll. Hal ini dilakukan agar tidak mempengaruhi kualitas model pada saat melakukan prediksi. Dampak penghapusan missing value terhadap model ini adalah untuk memastikan kualitas dan kinerja model machine learning yang optimal. Karena dataset yang dimiliki oleh proyek ini memiliki banyak missing value. Oleh karena itu penting untuk dilakukan penghapusan missing value. Penghapusan missing value dilakukan untuk semua fitur. Setelah dilakukan penghapusan sample data berkurang menjadi 2011. Ini merupakan penurunan yang sangat besar dan dapat menunjukkan juga bahwa data memiliki banyak missing value.
- Menghapus outlier

   Pada tahap ini, menganalisis distribusi data dan mengidentifikasi serta menghapus outlier dari variabel-variabel numeric. Hal ini dilakukan agar model dapat meningkatkan akurasi. Penghapusan outlier sangat penting untuk model karena dapat mempengaruhi kinerja model machine learning. Apalagi dataset yang dimiliki pada proyek ini merupakan data numeric. Jadi sangat penting untuk dilakukan penghapusan outlier. Teknik yang dipakai untuk menghapus outlier adalah dengan menggunakan IQR. Pada dasarnya teknik ini akan menghapus nilai yang berada di luar ambang batas. Penghapusan outlier dilakukan untuk semua variabel. Setelah dilakukan penghapusan sample data berubah menjadi 1794.
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.

  Pada tahap ini dataset akan menjadi dua bagian yaitu train set (untuk melatih model) dan test set (untuk menguji model). Ini membantu menilai sejauh mana model dapat umum digunakan pada data baru. Pembagian data yang terjadi pada proyek ini adalah 1435 untuk training dan 359 untuk testing.
- Normalisasi menggunakan StandardScaler

  Pada langkah ini, data akan dinormalisasi menggunakan StandardScaler agar setiap variabel memiliki skala yang serupa. Hal ini membantu untuk model melakukan prediksi. Langkah-langkah untuk melakukan normalisasi menggunakan standard scaler diantaranya yaitu:

    1. Import library
    2. Inisialisasi StandardScaler
    3. Fit data pelatihan
    4. Transform data pelatihan

    Setelah normalisasi data menjadi seragam dan dapat digunakan untuk melakukan prediksi.

## Modeling
Pada proses modeling proyek ini menggunakan 3 algoritma yaitu :
1. KNN

    KNN adalah metode klasifikasi berbasis instan yang bekerja dengan cara menemukan kelas mayoritas dari k tetangga terdekat suatu titik data yang belum diketahui kelasnya. KNN dipilih karena sifatnya yang non-parametrik dan mampu menangani pola kompleks dalam data tanpa membuat asumsi tertentu tentang distribusi data. Cocok untuk kasus ini karena dataset memiliki berbagai parameter kualitas air yang kompleks dan tidak memiliki asumsi tertentu. Parameter yang digunakan pada proyek ini adalah :

     - ```n_neighbors=7```. Parameter tersebut menunjukkan jumlah k tetangga terdekat.

    Kinerja model ini terhadap dataset yang ada cukup baik. Kenapa algoritma KNN dipilih untuk proyek ini adalah karena KNN tidak membuat asumsi tertentu tentang distribusi data. Karena jika dilihat dataset yang ada di proyek ini memiliki distribusi data yang heterogen atau tidak mengikuti distribusi tertentu. Oleh karena itu, ini algoritma ini diharapkan dapat bekerja dengan baik untuk dataset yang proyek ini.

2. Random forest

    Random Forest adalah metode ensemble yang menggabungkan beberapa model pohon keputusan untuk meningkatkan kinerja dan kestabilan prediksi. Random Forest dipilih karena sifatnya sebagai metode ensemble yang menggabungkan beberapa pohon keputusan. Mampu menangani data yang kompleks dan cenderung overfitting, sehingga dapat memberikan hasil yang lebih stabil dan akurat. Ini menjadikan model dapat memprediksi dataset dengan banyak parameter dan meningkatkan akurasi dari model. Cocok untuk kasus ini karena dataset dari proyek ini memiliki 10 parameter. Kinerja model ini terhadap dataset yang ada cukup baik. Alasan mengapa random forest dipilih sebagai algoritma  untuk proyek ini adalah karena algoritma random forest dapat menangani fitur yang banyak. Dengan 10 fitur yang ada, diharapkan algoritma ini mampu memberikan hasil prediksi yang akurat.

3. Naive bayes

    Naive Bayes adalah metode klasifikasi berbasis probabilitas yang menggunakan teorema Bayes dengan asumsi independensi antar fitur. Naive Bayes dipilih karena kesederhanaannya dan ketangguhannya dalam menangani data dengan dimensi tinggi serta dapat memberikan hasil yang baik bahkan dengan asumsi independensi yang naif. Cocok dengan dataset yang memiliki sejumlah variabel penentu seperti proyek ini. Kinerja model ini terhadap dataset kurang bagus hasilnya. Alasan mengapa algoritma naive bayes dipilih sebagai algoritma untuk proyek ini adalah karena kemampuannya menangani fitur yang independensi satu sama lain. Karena jika dilihat dari korelasi antar fitur nya, fitur cenderung memiliki relasi lemah antar satu sama lain. Oleh karena itu naive bayes dipilih untuk menangani masalah itu.

## Evaluation
Metrik evaluasi yang digunakan adalah akurasi. Karena ini merupakan masalah klasifikasi jadi menggunakan akurasi sebagai metrik evaluasi. Berdasarkan proses modeling didapat hasil:
| KNN | Random Forest | Naive Bayes |
|:--------------:|:--------------:|:--------------:|
| 56.54 | 59.05   | 40.94  |

Dapat dilihat dari tabel diatas bahwa algoritma random forest memiliki skor akurasi yang terbesar yaitu **59.05**. Skor akurasi Random Forest 59.05, artinya 59% dari total prediksi yang dilakukan oleh model Random Forest benar. Oleh karena itu random forest adalah algoritma yang akan digunakan untuk melakukan prediksi selanjutnya. Namun, untuk pengembangan proyek lebih lanjut, diperlukan evaluasi yang lebih mendalam terhadap kinerja model random forest ini. Meskipun skor akurasi dapat memberikan gambaran umum tentang seberapa baik model melakukan prediksi, beberapa metrik evaluasi tambahan perlu dipertimbangkan. Seperti precision, recall, dan F1-score. Karena score recall memiliki score yang lumayan tinggi maka model mungkin perlu menimbangkan recall sebagai metrik evaluasi. Seperti yang dapat dilihat dari tabel dibawah ini.
| KNN | Random Forest | Naive Bayes |
|:--------------:|:--------------:|:--------------:|
| 94.81| 100   | 0  |

Jika dilihat dari tabel diatas algoritma random forest memiliki score yang paling tinggi yaitu 100. Dapat dilihat juga bahwa naive bayes menghasilkan recall sebesar 0. Yang mengindikasikan alasan mengapa performa naive bayes kurang memuaskan. Adalah karena model ini gagal mengidentifikasi setiap instance positif dalam dataset. Recall yang rendah seperti ini dapat disebabkan oleh beberapa faktor, seperti distribusi kelas yang tidak seimbang atau kurangnya fitur yang dapat membedakan antara kelas. Hasil dari proyek ini dilihat berdasarkan:

- Konteks data: Data yang dimiliki telah berhasil menghasilkan prediksi yang diinginkan.
- Problem statement: Hasil dari proyek ini telah berhasil untuk menjawab pertanyaan yang ada di problem statement. Semua fitur dapat mempengaruhi kualitas dari air.
- Goals: Hasil dari proyek ini belum bisa menentukan fitur apa yang paling mempengaruhi kualitas air. Namun proyek ini telah mampu membuat model machine learning untuk memprediksi kualitas air.

Kesimpulannya walaupun proyek ini telah berhasil dalam menghasilkan prediksi yang diinginkan berdasarkan konteks data dan menjawab problem statement dengan membangun model machine learning untuk memprediksi kualitas air, namun belum dapat secara langsung menentukan fitur apa yang paling mempengaruhi kualitas air. Meskipun demikian, pencapaian menciptakan model machine learning adalah langkah positif yang dapat digunakan untuk memahami hubungan antara fitur-fitur tersebut dengan lebih mendalam pada tahap selanjutnya. Analisis lebih lanjut dapat dilakukan untuk mengidentifikasi pengaruh relatif dari setiap fitur terhadap prediksi kualitas air, sehingga dapat memberikan wawasan lebih lanjut terkait faktor-faktor yang paling signifikan.

## Referensi
[1] 	U. A. Dahlan, "Bahaya Konsumsi Air Tidak Bersih," lldikti5.kemendikbud.com, 25 July 2023. [Online]. Available: https://lldikti5.kemdikbud.go.id/home/detailpost/bahaya-konsumsi-air-yang-tidak-bersih. [Accessed 3 January 2024].

[2] 	adminweb, "Pengaruh Kualitas Air Minum terhadap Kesehatan Masyarakat," expertindo-training.com, 11 August 2023. [Online]. Available: https://expertindo-training.com/pengaruh-kualitas-air-minum-terhadap-kesehatan-masyarakat/. [Accessed 4 February 2024].

[3] 	A. Aris, "ekonomi.bisnis.com," Waduh! Pencemaran Air Ternyata Bisa Hambat Pertumbuhan Ekonomi Global, 21 August 2019. [Online]. Available: https://ekonomi.bisnis.com/read/20190821/9/1139261/waduh-pencemaran-air-ternyata-bisa-hambat-pertumbuhan-ekonomi-global. [Accessed 4 February 2024].
