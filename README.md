# Laporan Proyek Machine Learning - I Gede Teguh Permana

## Domain Kesehatan

Kemajuan pesat dalam Machine Learning memberikan peluang besar bagi manusia untuk menyelesaikan permasalahan kompleks melalui pemanfaatan teknologi komputasi. Dalam proyek ini, penulis berencana menggunakan Machine Learning untuk melakukan prediksi terhadap penyakit diabetes pada pasien.

Diabetes melitus (DM) merupakan penyakit atau gangguan metabolisme kronis yang memiliki berbagai penyebab. Penyakit ini ditandai dengan kadar gula darah yang mencapai atau melebihi 200 mg/dl, serta kadar gula darah puasa yang mencapai atau melebihi 126 mg/dl. DM juga disertai gangguan metabolisme karbohidrat, lemak, dan protein akibat gangguan fungsi insulin. Gangguan ini bisa terjadi karena produksi insulin oleh sel-sel beta Langerhans di pankreas yang tidak memadai atau karena kurangnya responsivitas sel-sel tubuh terhadap insulin Kemenkes RI. DM dikenal sebagai "pembunuh diam-diam" karena sering kali penderita tidak menyadari kondisinya hingga muncul komplikasi serius (Kemenkes RI, 2014) (https://p2ptm.kemkes.go.id/informasi-p2ptm/penyakit-diabetes-melitus). Penyakit ini dapat memengaruhi hampir semua sistem tubuh, mulai dari kulit hingga jantung, dan berpotensi menimbulkan komplikasi yang serius. Berdasarkan laporan dari World Health Organization (WHO), Diabetes Mellitus menjadi penyebab kematian keenam tertinggi di dunia.

Mengingat bahaya besar yang ditimbulkan oleh DM, penulis ingin memanfaatkan tiga model Machine Learning—yaitu KNN Classifier, Random Forest Classifier, dan Boost Classifier—untuk memprediksi penyakit ini pada pasien menggunakan dataset dari Kaggle (https://www.kaggle.com).


## Business Understanding
Berdasarkan uraian latar belakang yang telah dijelaskan sebelumnya, berikut adalah rincian permasalahan yang akan diselesaikan dalam proyek ini:
- Bagaimana membangun model machine learning untuk mengelompokkan pasien yang menderita diabetes dan yang tidak?
- Apa saja faktor penyebab yang berkontribusi pada terjadinya diabetes pada pasien?


### Goals
- Mengidentifikasi model dengan tingkat akurasi tinggi untuk memprediksi diabetes pada pasien.
- Menganalisis faktor-faktor yang berkontribusi terhadap risiko diabetes pada pasien.

    ### Solution statements
    - Melaksanakan Exploratory Data Analysis untuk mengidentifikasi data yang memiliki pengaruh signifikan terhadap pasien yang menderita diabetes.
    - Menerapkan beberapa model Machine Learning untuk memprediksi pasien yang terdiagnosis diabetes. Model yang akan digunakan meliputi:
      - *Support Vector Classifier*
      - *Random Forest Classifier*
      - *Bagging + Decision Tree Classifier*

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari platform Kaggle.com dan dipublikasikan oleh AKSHAY DATTATRAY KHARE. Dataset ini bersumber dari National Institute of Diabetes and Digestive and Kidney Diseases.

Tujuan utama dari dataset ini adalah untuk secara diagnostik memprediksi apakah seorang pasien mengidap diabetes berdasarkan sejumlah pengukuran diagnostik yang tercantum dalam data. Dataset ini terdiri dari satu file dalam format CSV.

### Informasi data:

**Sumber**: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

## Keterangan Informasi Attribute 

| Attribute                   | Keterangan                                                   |
|-----------------------------|--------------------------------------------------------------|
| **Pregnancies**             | resentasi jumlah kehamilan                           |
| **Glucose**                 | representasi tingkat glukosa dalam darah                |
| **BloodPressure**           | representasi pengukuran tekanan darah                   |
| **SkinThickness**           | representasi ketebalan kulit                            |
| **Insulin**                 | representasi tingkat insulin dalam darah                |
| **BMI**                     | representasi indeks massa tubuh                         |
| **DiabetesPedigreeFunction**| representasi persentase diabetes                        |
| **Age**                     | representasi umur                                       |
| **Outcome**                 | representasi hasil akhir: 1 adalah diabetes dan 0 adalah tidak diabetes |

Berkas tersebut berisi informasi tentang 768 pasien yang mencakup 9 kolom data, tanpa adanya nilai yang hilang (missing values) maupun data yang duplikat.

### Statistik Data Deskriptif

| Statistic            | Pregnancies | Glucose   | BloodPressure | SkinThickness | Insulin    | BMI       | DiabetesPedigreeFunction | Age       | Outcome   |
|-----------------------|-------------|-----------|---------------|---------------|------------|-----------|--------------------------|-----------|-----------|
| **Count**            | 768.000000  | 768.000000 | 768.000000    | 768.000000    | 768.000000 | 768.000000 | 768.000000              | 768.000000 | 768.000000 |
| **Mean**             | 3.845052    | 120.894531 | 69.105469     | 20.536458     | 79.799479  | 31.992578  | 0.471876                | 33.240885  | 0.348958  |
| **Standard Deviation** | 3.369578    | 31.972618  | 19.355807     | 15.952218     | 115.244002 | 7.884160   | 0.331329                | 11.760232  | 0.476951  |
| **Minimum**          | 0.000000    | 0.000000   | 0.000000      | 0.000000      | 0.000000   | 0.000000   | 0.078000                | 21.000000  | 0.000000  |
| **25th Percentile**  | 1.000000    | 99.000000  | 62.000000     | 0.000000      | 0.000000   | 27.300000  | 0.243750                | 24.000000  | 0.000000  |
| **Median (50%)**     | 3.000000    | 117.000000 | 72.000000     | 23.000000     | 30.500000  | 32.000000  | 0.372500                | 29.000000  | 0.000000  |
| **75th Percentile**  | 6.000000    | 140.250000 | 80.000000     | 32.000000     | 127.250000 | 36.600000  | 0.626250                | 41.000000  | 1.000000  |
| **Maximum**          | 17.000000   | 199.000000 | 122.000000    | 99.000000     | 846.000000 | 67.100000  | 2.420000                | 81.000000  | 1.000000  |


#### Interpretasi Deskripsi statistik data 

Pada kolom Glucose, BloodPressure, SkinThickness, Insulin, dan BMI terdapat nilai minimum yang bernilai 0. Nilai tersebut tidak realistis, karena manusia tidak dapat memiliki kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, atau BMI sebesar nol. Oleh karena itu, nilai nol pada kolom-kolom tersebut akan dihapus.


### Berikut Visualisasi data dengan Boxplot: <br>
<img src="Hasil\Boxplot.png" style="zoom:60%;" /> <br>

#### Interpretasi Outlier pada Boxplot.
- Dalam boxplot `Pregnancies`: Grafik memperlihatkan outlier untuk jumlah kehamilan 13.0, 15.0, 17.0
    - Tidak di hapus karena ada kemungkinan seorang ibu hamil dalam range 13-17 kali
- Dalam Boxplot `Glucose`: tampaknya terdapat glucose 0 sebagai outlier
    - Di hapus karena data tersebut outlier, tidak ada orang memiliki kadar glukosa 0
- Dalam Boxplot `Blood Pressure`: Grafik memperlihatkan outlier untuk 0,21,23, 110 - 120
    - Normal karena tekanan darah di bawah 30 itu termasuk hipotensi dan di 110-120 bisa di bilang normal
- Dalam Boxplot `SkinThickness`: Grafik memperlihatkan outlier dengan hasil sebesar 100
    - tidak di hapus karena ketebalan kulit seseorang bervariasi
- Dalam Boxplot `Insulin`: Kadar Insulin menunjukan fluktuasi yang cukup tinggi.
    - tidak di hapus karena kadar inslusin cukup bervariasi fluktuatif
- Dalam Boxplot: `BMI`: Beberapa data memiliki fluktuasi BMI cukup tinggi
    - Tidak di hapus karena BMI pada penderita diabetes emang di harapkan lebih tinggi karena kelebihan makan dan tidak ideal
- Dalam Boxplot: `DiabetesPedigreFunction`: Mengandung nilai yang bervariasi tergantung dari riwayat masing masing keluarga
    - Tidak di hapus karena riwayat keluarga tersebut bervariasi
- Dalam Boxplot: `Age`: terdapat outlier pada usia yg tua
    - Nilai pada usia diatas 65 emang ada untuk beberapa jadi tidak di hapus. <br><br><br>



### Berikut Visualisasi data Categorical Features pada plot : <br>
<img src="Hasil\cat1.png" style="zoom:70%;" /> 
<img src="Hasil\cat2.png" style="zoom:70%;" />
Pada grafik Pregnancies, terlihat bahwa jumlah kehamilan terbanyak adalah 1. Sementara itu, pada grafik Outcome, distribusi data penderita diabetes dan non-diabetes tidak seimbang, dengan jumlah data yang terkena diabetes lebih banyak dibandingkan yang tidak terkena diabetes. Dikarenakan Pregnancies tidak memiliki korelasi yang cukup signifikan maka akan di drop pada saat split data <br><br><br>



### Berikut Visualisasi data Numerical Features pada histogram :
<img src="Hasil\num.png" style="zoom:70%;" /><br> 

#### Interpretasi histogram
- Meaning Histogram:
    - Beberapa kolom berdistribusi right-skewed
    - Normaly distribution : Glucose, BloodPressure, SkinThickness, BMI


### Multivariate Analysis
Hubungan Numerical Features Terhadap Outcome.
<img src="Hasil\Glucose_Outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\BloodPressure_Outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\Age_Outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\BMI_outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\Skin_Outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\Pedigree_Outcome.png" style="zoom:70%;" /><br>
<img src="Hasil\Insulin_Outcome.png" style="zoom:70%;" /><br>

#### Interpertasi
- Meaning Plot:
    Pada data tersebut kita bisa melihat bahwa ciri ciri outcome (1/0) penderita diabetes memiliki kategori :
    - Pasien Representasi Glukosa Tinggi
    - Pasien Representasi BloodPressure Tinggi
    - Pasien Representasi BMI Tinggi
    - Pasien Representasi DiabetesPedigreeFunction Tinggi
    - Pasien Representasi Age Semakin Tua <br><br><br>


### Heat Map
Pada data numerik, digunakan heatmap untuk memvisualisasikan korelasi antara fitur 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', dan 'Age' dengan data "Outcome" agar lebih mudah dilihat dan dipahami. <br> <img src="Hasil\Heatmap_fitur.png" style="zoom:70%;" /><br> Hasil dari heatmap menunjukkan bahwa diabetes (outcome) memiliki korelasi yang signifikan terhadap Glucose, BMI, dan Age.


## Data Preparation

- Mengatasi data kosong <br>

| missing | value |
|---|----|
| Pregnancies | 0 |
| Glucose |	0 |
| BloodPressure | 0 |
| SkinThickness | 0 |
| Insulin |	0 |
| BMI |	0 |
| DiabetesPedigreeFunction | 0 |
| Age |	0 |
| Outcome |	0 |
<br>
  Tahapan ini bertujuan untuk mengetahui missing data 

- Balancing Dataset <br>
 <img src="Hasil\Balalancing.png" style="zoom:70%;" /><br> Pada tahap Balancing Dataset, diperlukan untuk menyeimbangkan data Outcome yang tidak seimbang. Tanpa keseimbangan, model cenderung memprioritaskan kategori dengan jumlah data yang lebih banyak. Oleh karena itu, tahap ini dilakukan dengan menggunakan teknik yang menghasilkan data dummy atau data sintetis.


- Membagi data menjadi data *training* dan *testing* <br>
  Tahap ini bertujuan agar model yang dilatih dapat diuji menggunakan data yang berbeda dari data yang digunakan selama pelatihan. Data dibagi menjadi dua bagian, yaitu training dan testing. Pembagiannya adalah 80% untuk training dan 20% sisanya untuk testing. Fungsi train_test_split dibuat from scracth. <br>



## Modeling
Algoritma pada *Machine Learning* yang digunakan antara lain : 
- **Support Vector Classifier (SVC)**, SVC bekerja dengan mencari hyperplane yang memisahkan kelas-kelas dalam data dengan margin terbesar. Dalam konteks klasifikasi, SVC mencoba menemukan garis atau bidang yang memisahkan dua kelas dengan jarak terbesar antara data dari kedua kelas tersebut. Pada penelitian ini, yang menjadi masalah klasifikasi adalah apakah pasien terkena diabetes atau tidak. Proyek ini menggunakan sklearn.svm.SVC dengan memasukkan X_train dan y_train untuk membangun model. Parameter yang digunakan pada proyek ini adalah C, yang mengontrol kompleksitas model. Nilai C yang lebih besar akan membuat model lebih fokus pada kesalahan klasifikasi, sedangkan nilai yang lebih kecil memberikan model fleksibilitas lebih besar untuk kesalahan, yang dapat menghindari overfitting.


- **Random Forest Classifier**, merupakan salah satu algoritma populer yang digunakan karena kesederhanaannya dan memiliki stabilitas yang baik. Proyek ini menggunakan [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=sklearn+ensemble+randomforestclassifier#sklearn.ensemble.RandomForestClassifier) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
     `criterion` = Fungsi untuk mengukur kualitas split.
     `n_estimators` = Jumlah tree pada forest.
     `max_depth` = Kedalaman maksimum setiap tree.
     `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi.
Pada model ini menggunakan parameter criterion = gini ,n_estimators=100, max_depth=9, random_state=44. 

- **Bagging with Decision Tree Classifier**, Bagging (Bootstrap Aggregating) adalah teknik ensemble yang melibatkan pelatihan beberapa model pada subset data yang berbeda, kemudian menggabungkan prediksi mereka untuk menghasilkan hasil akhir yang lebih stabil dan akurat. Dalam hal ini, model yang digunakan adalah Decision Tree Classifier, yang merupakan model pohon keputusan yang membagi data berdasarkan fitur untuk membuat keputusan. Proyek ini menggunakan sklearn.ensemble.BaggingClassifier dengan Decision Tree sebagai estimator. Model dibangun dengan memasukkan X_train dan y_train. Parameter yang digunakan dalam proyek ini termasuk n_estimators, yang menentukan jumlah pohon keputusan yang akan dilatih, dan max_samples, yang mengatur jumlah sampel data yang digunakan untuk setiap pohon keputusan. Bagging membantu mengurangi variansi model dengan melatih model-model independen pada data yang berbeda.



## Evaluation
Pada proyek ini model dibuat untuk mengklasifikasikan pasien terkena diabetes dan tidak terkena diabetes. Hasil evaluasi diperoleh bahwa model yang memiliki Performance tertinggi iyalah Random Forest Classifier & Bagging Classifier. <br>
<img src="Hasil\Performance_Model.png" style="zoom:70%;" /><br>

Sebelum menghitung Accuracy, Precision, Recall, dan F1-score. Akan dijelaskan mengenai *confusion matrix* terdapat empat nilai, yakni *true positive*,  *true negative*, *false positive* dan *false negative*.. <br>

<img src="Hasil\Keseluruhan.PNG" style="zoom:50%;" /><br>
Pada hasil dari keseluruhan dapat dilihat bahwa model pada RF dan Bagging mengalami *overfitting*, emang secara akurasi lebih tinggi namun dalam keadaan good fit, SVC tampak lebih good fit dibandingkan model RF dan Bagging. <br>

### a. SVC
Berikut hasil dari confusion matriks dari model SVC sebagai good fit dibandingkan model RF dan Bagging. <br>
<img src="Hasil\CM_SVC.png" style="zoom:70%;" /><br>

- *Accuracy*

  Metric akurasi mengukur sejauh mana nilai prediksi mendekati nilai aktual. Untuk menghitungnya, cukup dengan membagi jumlah prediksi yang benar dengan total data. Akurasi cocok digunakan pada kasus dengan data yang seimbang.

- *Precision*

  Metric ini mengukur tingkat ketepatan antara informasi yang diminta oleh pengguna dan jawaban yang diberikan oleh sistem. Nilai precision dapat dihitung menggunakan rumus yang tertera di bawah ini.   *Metric* ini fokus pada kinerja model dalam memprediksi label data positif. <br>

- Recall

  Metric ini mengukur tingkat keberhasilan sistem dalam menemukan kembali informasi yang hilang. Nilai recall dapat dihitung dengan rumus di bawah ini. Berbeda dengan precision yang hanya memperhitungkan label positif, metric ini menghitung bagian negatif dari prediksi label positif.<br>

- F1-score

  Metric ini adalah rata-rata harmonik antara precision dan recall. Nilai f1-score dapat dihitung dengan rumus di bawah ini. Selanjutnya model Random Forest Classifier, akan dihitung *metrics* f1-score dan recall. <br>

Adapun hasil yang diperoleh pada model SVC sebagai model yang good fit sebagai berikut pada data testing:
Akurasi SVC : 0.7721518987341772 <br>
----------------------------------------------------------------------------------------------------
Precision SVC : 0.7222222222222222 <br>
----------------------------------------------------------------------------------------------------
Recall SVC : 0.5 <br>
----------------------------------------------------------------------------------------------------
F1-score SVC : 0.5909090909090909 <br>

Berdasarkan Projek ini , hasil metric yang diperoleh pada SVC lebih condong ke precision dibandingkan recall maka dari itu model SVC dapat di evaluasi untuk mengatasi masalah Recall.

### b. RF
Berikut hasil dari confusion matriks dari model RF yang mana lebih overfitting dibandingkan model SVC. <br>
<img src="Hasil\CM_RF.png" style="zoom:70%;" /><br>
- *Accuracy*

  Metric akurasi mengukur sejauh mana nilai prediksi mendekati nilai aktual. Untuk menghitungnya, cukup dengan membagi jumlah prediksi yang benar dengan total data. Akurasi cocok digunakan pada kasus dengan data yang seimbang.

- *Precision*

  Metric ini mengukur tingkat ketepatan antara informasi yang diminta oleh pengguna dan jawaban yang diberikan oleh sistem. Nilai precision dapat dihitung menggunakan rumus yang tertera di bawah ini.   *Metric* ini fokus pada kinerja model dalam memprediksi label data positif. <br>

- Recall

  Metric ini mengukur tingkat keberhasilan sistem dalam menemukan kembali informasi yang hilang. Nilai recall dapat dihitung dengan rumus di bawah ini. Berbeda dengan precision yang hanya memperhitungkan label positif, metric ini menghitung bagian negatif dari prediksi label positif.<br>

- F1-score

  Metric ini adalah rata-rata harmonik antara precision dan recall. Nilai f1-score dapat dihitung dengan rumus di bawah ini. Selanjutnya model Random Forest Classifier, akan dihitung *metrics* f1-score dan recall. <br>

Adapun hasil yang diperoleh pada model RF sebagai model yang overfitting sebagai berikut pada data testing:
Akurasi RF : 0.7721518987341772 <br>
----------------------------------------------------------------------------------------------------
Precision RF : 0.7 <br>
----------------------------------------------------------------------------------------------------
Recall RF :  0.5384615384615384 <br>
----------------------------------------------------------------------------------------------------
F1-score RF : 0.6086956521739131 <br>


### c. Bagging + Decision Tree Classifier
Berikut hasil dari confusion matriks dari model Bagging + Decision Tree Classifier yang mana lebih overfitting dibandingkan model SVC. <br>
<img src="Hasil\CM_BAGGING.png" style="zoom:70%;" /><br>

- *Accuracy*

  Metric akurasi mengukur sejauh mana nilai prediksi mendekati nilai aktual. Untuk menghitungnya, cukup dengan membagi jumlah prediksi yang benar dengan total data. Akurasi cocok digunakan pada kasus dengan data yang seimbang.

- *Precision*

  Metric ini mengukur tingkat ketepatan antara informasi yang diminta oleh pengguna dan jawaban yang diberikan oleh sistem. Nilai precision dapat dihitung menggunakan rumus yang tertera di bawah ini.   *Metric* ini fokus pada kinerja model dalam memprediksi label data positif. <br>

- Recall

  Metric ini mengukur tingkat keberhasilan sistem dalam menemukan kembali informasi yang hilang. Nilai recall dapat dihitung dengan rumus di bawah ini. Berbeda dengan precision yang hanya memperhitungkan label positif, metric ini menghitung bagian negatif dari prediksi label positif.<br>

- F1-score

  Metric ini adalah rata-rata harmonik antara precision dan recall. Nilai f1-score dapat dihitung dengan rumus di bawah ini. Selanjutnya model Random Forest Classifier, akan dihitung *metrics* f1-score dan recall. <br>

Adapun hasil yang diperoleh pada model Bagging + Decision Tree Classifier sebagai model yang overfitting sebagai berikut pada data testing:
Akurasi Bagging : 0.759493670886076 <br>
----------------------------------------------------------------------------------------------------
Precision Bagging : 0.6521739130434783 <br>
----------------------------------------------------------------------------------------------------  
Recall Bagging :  0.5769230769230769 <br>
----------------------------------------------------------------------------------------------------
F1-score Bagging : 0.6122448979591837 <br>

## Kesimpulan
Dari proyek prediksi pasien diabetes dan non-diabetes menggunakan tiga model Machine Learning, yaitu Support Vector Classifier, Random Forest Classifier, dan Bagging with Decision Tree Classifier, dapat disimpulkan bahwa algoritma `Support Vector Classifier` menunjukkan performa terbaik dibandingkan yang lain. Hal ini terlihat dari model yang lebih sesuai (good fit) dibandingkan dengan algoritma lainnya yang cenderung mengalami overfitting.

Dalam proyek ini, kami memperoleh informasi tentang ciri-ciri atau faktor yang membuat pasien menderita diabetes, dengan kriteria sebagai berikut:

- Pasien dengan kadar glukosa tinggi
- Pasien dengan tekanan darah tinggi
- Pasien dengan BMI tinggi
- Pasien dengan DiabetesPedigreeFunction tinggi
- Pasien yang lebih tua (Age)


# Referensi

1. [Dicoding](https://www.dicoding.com/academies/319/tutorials/16979?from=17053) (2021). *Machine learning Terapan*
2. [Imbalanced-learn](https://imbalanced-learn.org/stable/). *Documentation*
3. [Kemenkes RI](https://p2ptm.kemkes.go.id/informasi-p2ptm/penyakit-diabetes-melitus)
4. [World Health Organization](https://www.who.int/health-topics/diabetes)
5. [Kaggle](https://www.kaggle.com) 
