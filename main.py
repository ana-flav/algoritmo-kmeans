import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns


dataset = pd.read_csv('Mall_Customers.csv')


X = dataset.iloc[:, [3, 4]].values
plt.subplot(1, 2, 1)
sns.histplot(dataset['Annual Income (k$)'], kde=True, bins=20, color='blue')
plt.title('Distribuição de Annual Income (k$)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequência')

plt.subplot(1, 2, 2)
sns.histplot(dataset['Spending Score (1-100)'], kde=True, bins=20, color='green')
plt.title('Distribuição de Spending Score (1-100)')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

silhouette_scores = []
for i in range(2, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)


plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='blue')
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()


optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  
print(f"Optimal number of clusters: {optimal_k}")


kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0)
y_kmeans = kmeans_model.fit_predict(X)


dataset['Cluster'] = y_kmeans

plt.figure(figsize=(10, 7))
cluster_labels = {
    0: 'Avarage Income, Avarage Spending',
    1: 'Avarage Income, High Spending',
    2: 'Average Income, Low Spending',
    3: 'Low Income, Low Spending',
    4: 'Low Income, High Spending'
}

for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=cluster_labels[i])

plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], 
            s=300, c='black', marker='*', label='Centroids')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
cluster_stats = dataset.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("Médias por Cluster:")
print(cluster_stats)

dataset['Cluster Label'] = dataset['Cluster'].map(cluster_labels)


print(dataset[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', 'Cluster Label']].head())


print("\nResumo por Cluster:")
print(dataset.groupby('Cluster Label')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

cluster_counts = dataset['Cluster Label'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title('Número de Clientes em Cada Cluster Nomeado')
plt.xlabel('Cluster')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=45)
plt.grid()
plt.show()
