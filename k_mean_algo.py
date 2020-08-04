
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans

dataset = pd.read_csv('D:\PythonCode\dataset\konsumen.csv')
print(dataset)
dataku= pd.DataFrame(dataset)


#array
x= np.array(dataset)
print(x)

#scater plot
plt.scatter(x[:,1],x[:,2], label = 'True Position')
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('scater Gaji vs Pengeluaran')
plt.show()
#fiting
kmeans= KMeans(n_clusters=5)
kmeans.fit(x)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

#scater plot kmeans
plt.scatter(x[:,1],x[:,2], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('scater Gaji vs Pengeluaran')
plt.show()

#scater plot centroid
plt.scatter(x[:,1],x[:,2], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],color='black')
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('scater Gaji vs Pengeluaran')
plt.show()

print('hallo git i love u')