import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ast
from puli import df1

consi = df1[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'genres']]
audio_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

consi['genres_top'] = df1['genres'].apply(ast.literal_eval)
consi['genres_top'] = consi['genres_top'].apply(lambda x: x[0] if x else None)

consi = consi.drop(['genres'], axis=1)

# Scale Tempo
consi['tempo'] = MinMaxScaler().fit_transform(consi['tempo'].values.reshape(-1,1))

# Normalize audio columns
consi[audio_cols] = Normalizer().fit_transform(consi[audio_cols])

# Average features by genre
genres_avg = consi.groupby('genres_top').mean()

# Standardize the genre features
X_scaled = StandardScaler().fit_transform(genres_avg)

# Elbow method for optimal clusters
Ks = np.arange(1, 20)
score = [-KMeans(n_clusters=i, random_state=1986).fit(X_scaled).score(X_scaled) for i in Ks]

plt.figure(figsize=(10,10))
plt.plot(Ks, score)
plt.xlim(0, 18)
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Error')
plt.title('Elbow Method')
#plt.show()

# Apply PCA for visualization
optimal_clusters = 10  # this should be chosen based on the Elbow Method
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

genres_avg['x'] = X_pca[:,0]
genres_avg['y'] = X_pca[:,1]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=1986)
genres_avg['cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(12,10))
plt.scatter(genres_avg['x'], genres_avg['y'], c=genres_avg['cluster'], cmap='viridis', s=50, alpha=0.6)
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title(f'2D PCA of Genres with {optimal_clusters} Clusters')
plt.legend()
plt.grid(True)
#plt.show()

# Parallel Coordinates plot for cluster centroids
centroids = kmeans.cluster_centers_
df_centroids = pd.DataFrame(centroids, columns=audio_cols)
df_centroids['cluster'] = df_centroids.index  # add cluster index for coloring

plt.figure(figsize=(12,6))
pd.plotting.parallel_coordinates(df_centroids, class_column='cluster', color=plt.cm.jet(np.linspace(0, 1, optimal_clusters)), linewidth=2)
plt.title('Parallel Coordinates of Cluster Centroids')
plt.grid(True)
plt.show()
