# dépot github : https://github.com/NorbertSourou/kmeans-python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [1,1,2,2,3,3,4,4,5,5,6,7]
y = [7,6,6,5,6,4,1,2,1,2,1,3]
myDic= {"ROI": x, "Dette": y}

# Si vous voulez uploader un fichier décommenter la ligne suivante  et commenter la deuxième

# data = pd.read_csv("chemin_du_fichier", sep=';')

data = pd.DataFrame(data= myDic)
print(data)


# Sélections de k point : les centroides
k = 2 # nombre de clusters

# Point calculer  ramdoms
centers = np.random.uniform(low=data.min().values, high=data.max().values, size=(k, 2))

# Points choisis  ramdoms
#centers = data.sample(n=k, random_state=0).values
print(centers)

# distance euclidienne de clusters
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# distance euclidienne de clusters
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


# assignation  des clusters
def assign_clusters(data, centers):
    clusters = np.zeros(len(data))
    for i in range(len(data)):
        distances = [manhattan_distance(data.iloc[i].values, center) for center in centers]
        clusters[i] = np.argmin(distances)
    print(clusters)
    return clusters


# calcule  des clusters
def update_centers(data, clusters, centers):
    new_centers = np.zeros_like(centers)
    for i in range(len(centers)):
        if np.sum(clusters == i) > 0:
            new_centers[i] = np.mean(data[clusters == i], axis=0)
        else:
            new_centers[i] = centers[i]
    return new_centers

# Vérification pour  des clusters
max_iter = 100 # nombre maximum d'itérations
for i in range(max_iter):
    old_centers = centers
    clusters = assign_clusters(data, centers)
    centers = update_centers(data, clusters, centers)
    if np.allclose(old_centers, centers):
        break

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=300, c='red')
plt.show()