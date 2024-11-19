#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#Working with Dataset
customer_data = pd.read_csv('Customers.csv')
#customer_data.head()
#customer_data.info()
customer_data.isnull().sum()
X = customer_data.drop(columns=['CustomerID','Gender','Age'], axis=1).values
#Visualize the data points
plt.figure(figsize=(15,8))
sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.show()
#Find K value using elbow method
#we have used the KMeans class of sklearn. cluster library to form the clusters.
#Next, we have created the wcss_list variable to initialize an empty list, 
#which is used to contain the value of wcss computed for different values of k ranging from 1 to 10.
wcss=[]
for i in range(1,10):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=2)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
#Plot a line plot between WCSS and k
plt.figure(figsize=(15,8))
plt.plot(range(1,10), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('wcss')
plt.show()

#Model Training
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
#Centroid points
kmeans.cluster_centers_
#Visualize the clusters formed
plt.figure(figsize=(15,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='pink', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='green', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='orange', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
#In above lines of code, we have written code for each clusters, 
#ranging from 1 to 5. The first coordinate of the mtp.scatter, 
#i.e., x[y_predict == 0, 0] containing the x value for the showing the matrix of features values, 
#and the y_predict is ranging from 0 to 1.
