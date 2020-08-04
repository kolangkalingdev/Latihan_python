import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

bunga= datasets.load_iris()

#menampilkan type data dari bunga
print(type(bunga))
#menampilkan dictionary key dari bunga
print(bunga.keys())

#menampilkan type obyek dari atribut yang ada
print(type(bunga.data),type(bunga.target))
#menampilkan jumlah baris dan kolom
print(bunga.data.shape)
#menampilkan target set dari data
print(bunga.target_names)


#memanggil train data
X = bunga.data
#memanggil target data
Y= bunga.target


df = pd.DataFrame(X, columns=bunga.feature_names)

print(df.head())

#Memanggil KNN Classifier
knn= KNeighborsClassifier(n_neighbors=6,weights='uniform',algorithm='auto',metric='euclidean')

X_train = bunga['data']
y_train = bunga['target']

knn.fit(X_train,y_train)


#melakukan prediksi
data_test= [[6.2,1.5,4.3,2.7],[4.7,3.2,2.7,8.9],[5.3,5.1,7.1,10.1]]

Y_pred=knn.predict(data_test)

print("Hasil Prediksi: jenis bunga (0 = Sentosa, 1= versicolor, 2= virginica",Y_pred)


#Visualisasi Klasifikasi PERBAIKAN

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors


X=bunga.data[:, :2]
y=bunga.target


#mengatur warna 
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
#plot decision boundary
knn.fit(X,y)
x_min, x_max = X[:,0].min()-1,X[:,0].max()+1
y_min, y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max),np.arange(y_min,y_max))

A=knn.predict(np.c_[xx.ravel(), yy.ravel()])

#memasukkan hasil ke dalam color plot
z = A.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)

#plot training points
plt.scatter(X[:,0],X[:,1],c=y, cmap=cmap_bold,edgecolors='k',s=20)
plt.xlim(xx.min(),xx.max())
plt.xlim(yy.min(),yy.max())

plt.title('Klasifikasi Bunga Iris')

#Ploting
plt.show()
print('hallo git i love u')


















