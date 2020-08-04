import pandas as pd 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Memanggil dataset dari lib sklearn
bunga=datasets.load_iris()
#training dataset
X=bunga.data
Y=bunga.target

df = pd.DataFrame(X, columns=bunga.feature_names)
print(df.head())

print(bunga.data.shape)
print(bunga.target_names)

X_train,X_test,y_train,y_test =train_test_split(X,Y)

gnb= GaussianNB()
gnb.fit(X,Y)

print(gnb.predict(X))

data = [[6.2,1.5,4.2,2.6]]

Y_prediksi= gnb.predict(data)

print('Prediksi :jenis bunga',Y_prediksi)

#menghitung akurasi

print("accuracy =%0.2f"% accuracy_score(y_test,gnb.predict(X_test)))

print(classification_report(y_test, gnb.predict(X_test)))



