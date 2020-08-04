from sklearn import datasets 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 

bunga = datasets.load_iris()
print(dir(bunga))
print(bunga.feature_names)
print(bunga.target_names)
print(bunga.data)

x_axis = bunga.data[:, 0] #sepal length
y_axis = bunga.data[:, 1] #sepal width

plt.scatter(x_axis, y_axis, c=bunga.target)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('scater iris')
plt.show()

#activated dbscan
dbscan= DBSCAN()

dbscan.fit(bunga.data)
#transformasi menggunakan pca 2 D

pca =PCA(n_components=2).fit(bunga.data)
pca_2d =pca.transform(bunga.data)


#visualisasi
label = {0:'red',1:'blue',2:'green'}
print(label)

for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        cluster1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='^')
    elif  dbscan.labels_[i] == 1:
        cluster2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='.')
    elif dbscan.labels_[i] == -1:
        noise = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='x')
plt.legend([cluster1, cluster2, noise])
plt.title('Klasterisasi DBSCAN')
plt.show()


