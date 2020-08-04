import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
dataset=pd.read_csv('D:\PythonCode\dataset\Sales_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values
print(dataset.keys())
print(dataset.shape)
dataku = pd.DataFrame(dataset)
dataku.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred_train = regressor.predict(X_train)
print(y_pred_train)
y_pred_test = regressor.predict(X_test)
print(y_pred_test)
plt.scatter(dataku.BiayaPromo, dataku.NilaiPenjualan)
plt.xlabel("Biaya Promo")
plt.ylabel("Nilai Penjualan")
plt.title("Scatter Plot Promo vs Sales")

plt.show()

#VISUALASISASI HASIL PREDIKSI PADA DATA TRAIN
#ukuran Plot
plt.figure(figsize=(10,8))
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train,y_pred_train, color= 'red')


plt.title('Scatter Plot Promo vs Sales [Train set]')
plt.xlabel("Biaya Promo")
plt.ylabel("Nilai Penjualan")
plt.show()

#VISUALASISASI HASIL PREDIKSI PADA DATA Test
#ukuran Plot
plt.figure(figsize=(10,8))
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test,y_pred_test, color= 'red')


plt.title('Scatter Plot Promo vs Sales [Test set]')
plt.xlabel("Biaya Promo")
plt.ylabel("Nilai Penjualan")
plt.show()




