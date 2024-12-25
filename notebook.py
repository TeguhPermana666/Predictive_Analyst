# 1. Import Library
import os
from tqdm import tqdm
import zipfile
import wget
import opendatasets
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# 2. Load Dataset
os.environ['Your Kaggle username:'] = 'teguhpermana'
os.environ['Your Kaggle key:'] = '70dafea2a31376afc04704d9a2e705a5'
if os.path.exists('./diabetes-dataset/diabetes.csv'):
    print('File already exists')
else:
    opendatasets.download_kaggle_dataset(dataset_url='https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset', data_dir='.')
    
df = pd.read_csv(r'diabetes-dataset/diabetes.csv')


# 3. Data Understanding
"""
Sumber**: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

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

"""
def data_understanding(df):
    print('\nData Info:\n')
    df.info()
    print('\nJumlah Baris : ', df.shape[0])
    print('\nJumlah Kolom  : ', df.shape[1])
    print('\nMissing Value:\n', pd.DataFrame({'Missing Value': df.isnull().sum()}))
    print('\nDuplicate Data:\n', df[df.duplicated()])
    print('\nStatistik Data:\n', df.describe())
data_understanding(df)

# 4. EDA
## 4.1. Outlier Checking
out_cols = ['Pregnancies', 'Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
plt.figure(figsize=(16,13))
for i in range(len(out_cols)):
    plt.subplot(8,1,i+1)
    sns.boxplot(x=df[out_cols[i]])
    plt.tight_layout()    
    plt.title('Boxplot {}'.format(out_cols[i]))
    
"""
### Interprestasi Hasil Outlier Checking Boxplot Visualize
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
    - Nilai pada usia diatas 65 emang ada untuk beberapa jadi tidak di hapus.
"""

## 4.2. Delete Outlier
df = df.loc[(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]!=0).all(axis=1)]

# 5. Univariate Analysis
num_fitur = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
cat_fitur = ['Pregnancies', 'Outcome']

## 5.1. Categorical Features
### a. Pregnancies
fitur_pregnancies = cat_fitur[0]
count_fitur = df[fitur_pregnancies].value_counts()
percent_fitur = 100 * df[fitur_pregnancies].value_counts(normalize=True)
df_pregnancies = pd.DataFrame({'Count Fitur': count_fitur, 'Percent': percent_fitur.round(2)})
print(df_pregnancies)
count_fitur.plot(kind='bar', title=fitur_pregnancies)

### b. Outcome
fitur_outcome = cat_fitur[1]
count_outcome = df[fitur_outcome].value_counts()
percent_fitur = 100 * df[fitur_outcome].value_counts(normalize=True)
df_outcome = pd.DataFrame({'Count Fitur': count_outcome, 'Percent': percent_fitur.round(2)})
print(df_outcome)
count_outcome.plot(kind='bar', title=fitur_outcome)

## 5.2. Numerical Features
plt.figure(figsize= (22,20))
for i in range(len(num_fitur)):
    plt.subplot(len(num_fitur), 1, i+1)
    sns.histplot(x=df[num_fitur[i]])
    plt.title('Histogram Plot {}'.format(num_fitur[i]))
    plt.tight_layout()
"""
- Meaning Histogram:
    - Beberapa kolom berdistribusi right-skewed
    - Normaly distribution : Glucose, BloodPressure, SkinThickness, BMI
"""

# 6. Multivariate Analysis
## 6.1. Korelasi numerical fitur dengan outcome
for i in range(len(num_fitur)):
    plt.figure(figsize=(20,40))
    plt.subplot(len(num_fitur), 1, i+1)
    sns.kdeplot(x=df[num_fitur[i]], hue='Outcome', data=df)
    plt.title('Distribusi {} Dengan Outcome'.format(num_fitur[i]))
    plt.show()
    print('\n\n')
    
"""
- Meaning Plot:
    Pada data tersebut kita bisa melihat bahwa ciri ciri outcome (1/0) penderita diabetes memiliki kategori :
    - Pasien Representasi Glukosa Tinggi
    - Pasien Representasi BloodPressure Tinggi
    - Pasien Representasi BMI Tinggi
    - Pasien Representasi DiabetesPedigreeFunction Tinggi
    - Pasien Representasi Age Semakin Tua
"""

plt.figure(figsize=(12,10))
cor_matrix = df.corr().round(2)
sns.heatmap(data=cor_matrix, annot=True, cmap='coolwarm', linewidths=0.6)
plt.title('Correlation with Fitur Numerical', size=25)

# 7. Data Preparation
normal = df[(df['Outcome'] == 0)]
diabetes = df[(df['Outcome'] == 1)]

diabetes_upsample = sklearn.utils.resample(diabetes, 
                                         replace=True, 
                                         n_samples=len(normal), 
                                         random_state=42)

new_df = pd.concat([diabetes_upsample, normal])
sns.countplot(x=new_df['Outcome'])

def train_test_split(data, random_state=100, train_size = 0.8, ):
    # Shuffle the data
    np.random.seed(random_state)  
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data.iloc[shuffled_indices].reset_index(drop=True)

    # Split the data into train and test
    split_idx = int(len(shuffled_data) * train_size)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    # Separate features (X) and target (y) for train and test sets
    X_train = train_data.drop(columns=['Outcome','Pregnancies'])
    y_train = train_data["Outcome"]
    
    X_test = test_data.drop(columns=['Outcome','Pregnancies'])
    y_test = test_data["Outcome"]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(df, train_size=0.8, random_state=100)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 8. Modeling
"""
Tahapan modeling dilakukan dengan 2 metode: `Baseline Model` dan `Ensemble Model`. 
- Pada `Baseline Model` menggunakan : SVC 
- pada `Ensemble Model` menggunakan : RF dan Bagging
"""

## 8.1. SVC
from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)

## 8.2. RF
# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestClassifier

# buat model prediksi
RF = RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=9,random_state=42) 
RF.fit(X_train, y_train)

## 8.3. Bagging + Decision Tree Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=44)
bagging.fit(X_train, y_train)

# 9. Model Evaluation
list_model = [svc, RF, bagging]
model_names = ['Support Vector Classifier', 'Random Forest Classifier', 'Bagging Classifier (Decision Tree)']
scores = []

for i in list_model:
    score = sklearn.model_selection.cross_val_score(i, X_train, y_train, cv=5)
    score = np.mean(score)
    scores.append(score)
    
# Plot Performances Models
plt.figure(figsize=(9, 6))
barplot = sns.barplot(x=model_names, y=scores, hue=model_names, dodge=False, palette='viridis', legend=False)

for index, score in enumerate(scores):
    plt.text(index, score + 0.01, f"{score:.2f}", ha='center', fontsize=10, color='black')

plt.title("Performance Models")
plt.ylabel("Accuracy Score")
plt.xlabel("Model Names")
plt.ylim(0, 1)
plt.show()

print('Train RandomForestClassifierModel  : ' , RF.score(X_train, y_train))
print('Test RandomForestClassifierModel : ' , RF.score(X_test, y_test))

print('Train SVC  : ' , svc.score(X_train, y_train))
print('Test SVC : ' , svc.score(X_test, y_test))

print('Train Bagging  : ' , bagging.score(X_train, y_train))
print('Test Bagging : ' , bagging.score(X_test, y_test))

"""
Pada hasil yang di peroleh, di dapatkan bahwa model dengan ensemble method:
- RF dan Bagging didapatkan bahwa overfitting dengan hasil pada train sangat jauh marginnya dengan hasil pada test
- Maka dari itu disini menggunakan `SVC` yang lebih good fit walaupun secara performa diatas diperoleh lebih kecil daripada RF yakni 77%.
"""

y_pred_svc = svc.predict(X_test)
cm_svc = sklearn.metrics.confusion_matrix(y_test,y_pred_svc)

ax=sns.heatmap(cm_svc,annot=True)
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.title("Confusion Matrix untuk SVC ")
plt.show()

y_pred_bag = bagging.predict(X_test)
cm_bag = sklearn.metrics.confusion_matrix(y_test,y_pred_bag)

ax=sns.heatmap(cm_bag,annot=True)
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.title("Confusion Matrix untuk Bagging ")
plt.show()

y_pred_rf = RF.predict(X_test)
cm_rf = sklearn.metrics.confusion_matrix(y_test,y_pred_rf)

ax=sns.heatmap(cm_rf,annot=True)
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.title("Confusion Matrix untuk RF ")
plt.show()

test_score_svc = sklearn.metrics.accuracy_score(y_test,y_pred_svc)
print("Akurasi SVC : {}".format(test_score_svc))
print("-" * 100)

precision_score_svc = sklearn.metrics.precision_score(y_test,y_pred_svc)
print("Precision SVC : {}".format(precision_score_svc))
print("-" * 100)

recall_score_svc = sklearn.metrics.recall_score(y_test,y_pred_svc)
print("Recall SVC : {}".format(recall_score_svc))
print("-" * 100)

f1_score = sklearn.metrics.f1_score(y_test,y_pred_svc)
print("F1-score SVC : {}".format(f1_score))


test_score_bag = sklearn.metrics.accuracy_score(y_test,y_pred_bag)
print("Akurasi Bagging : {}".format(test_score_bag))
print("-" * 100)

precision_score_bag = sklearn.metrics.precision_score(y_test,y_pred_bag)
print("Precision Bagging : {}".format(precision_score_bag))
print("-" * 100)

recall_score_bag = sklearn.metrics.recall_score(y_test,y_pred_bag)
print("Recall Bagging : {}".format(recall_score_bag))
print("-" * 100)

f1_score = sklearn.metrics.f1_score(y_test,y_pred_bag)
print("F1-score Bagging : {}".format(f1_score))

test_score_rf = sklearn.metrics.accuracy_score(y_test,y_pred_rf)
print("Akurasi RF : {}".format(test_score_rf))
print("-" * 100)

precision_score_rf = sklearn.metrics.precision_score(y_test,y_pred_rf)
print("Precision RF : {}".format(precision_score_rf))
print("-" * 100)

recall_score_rf = sklearn.metrics.recall_score(y_test,y_pred_rf)
print("Recall RF : {}".format(recall_score_rf))
print("-" * 100)

f1_score = sklearn.metrics.f1_score(y_test,y_pred_rf)
print("F1-score RF : {}".format(f1_score))