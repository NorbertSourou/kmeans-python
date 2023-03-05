import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Définir la distance de Manhattan
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

# Définir la fonction pour attribuer chaque point au cluster le plus proche en utilisant la distance de Manhattan
def assign_clusters(data, centers):
    clusters = np.zeros(len(data))
    for i in range(len(data)):
        distances = [manhattan_distance(data.iloc[i].values, center) for center in centers]
        clusters[i] = np.argmin(distances)
    return clusters

# Définir la fonction pour mettre à jour les centres de cluster en utilisant la moyenne des points appartenant à chaque cluster
def update_centers(data, clusters, centers):
    new_centers = np.zeros_like(centers)
    for i in range(len(centers)):
        if np.sum(clusters == i) > 0:
            new_centers[i] = np.mean(data[clusters == i], axis=0)
        else:
            new_centers[i] = centers[i]
    return new_centers

# Définir la fonction pour effectuer le clustering
def kmeans_clustering(data, k):
    # Initialiser les centres de cluster de manière aléatoire
    centers = np.random.uniform(low=data.min().values, high=data.max().values, size=(k, 2))

    # Répéter les étapes de l'algorithme KMeans jusqu'à convergence
    max_iter = 100 # nombre maximum d'itérations
    for i in range(max_iter):
        old_centers = centers
        clusters = assign_clusters(data, centers)
        centers = update_centers(data, clusters, centers)
        if np.allclose(old_centers, centers):
            break

    # Retourner les clusters et les centres de cluster
    return clusters, centers

# Définir la fonction pour afficher les résultats du clustering
def plot_clusters(data, clusters, centers):
    # Afficher la représentation graphique de l'ensemble de points
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters)

    # Afficher les centres de cluster
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=300, c='red')

    # Afficher le graphique
    st.pyplot()

# Définir la fonction pour exécuter l'application Web
def main():
    # Titre de la page Web
    st.title("KMeans Clustering")

    # Télécharger le fichier de données
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(file, sep=';')  # spécifier le séparateur ';'

        # Convertir les valeurs de la deuxième colonne en type float64
        data['Dette'] = data['Dette'].astype('float64')

        # Sélectionner le nombre de clusters
        k = st.slider("Sélectionner le nombre de clusters", min_value=2, max_value=10)

        # Effectuer le clustering
        clusters, centers = kmeans_clustering(data, k)

        # Afficher les résultats
        plot_clusters(data, clusters, centers)

if __name__ == '__main__':
    main()