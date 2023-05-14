import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class SpectralClustering:
    def __init__(self, n_clusters, n_neighbors, affinity):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.affinity = affinity

    def fit(self, X):
        knn = KNeighborsClassifier(n_neighbors = self.n_neighbors, metric = 'euclidean')
        knn.fit(X)
        A = knn.predict_proba(X)
        if self.affinity == 'rbf':
            A = np.exp(-(A**2)/2)
        elif self.affinity == 'knn':
            A = np.maximum(A, A.T)
        else:
            print('Unknown affinity type.')
        D = np.diag(np.sum(A, axis = 1))
        L = D - A
        eigvals, eigvecs = np.linalg.eigh(L)
        sorted_indices = np.argsort(eigvals)[:self.n_clusters]
        cluster_assignments = eigvecs[:, sorted_indices]
        kmeans_model = KMeans(n_clusters = self.n_clusters)
        kmeans_model.fit(cluster_assignments)
        cluster_centers = kmeans_model.cluster_centers_
        predicted_labels = kmeans_model.predict(cluster_assignments)
        return predicted_labels
    
    def predict(self, X):
        A = self.knn.predict_proba(X)
        if self.affinity == 'rbf':
            A = np.exp(-(A**2)/2)
        elif self.affinity == 'knn':
            A = np.maximum(A, A.T)
        else:
            print('Unknown affinity type.')
        D = np.diag(np.sum(A, axis = 1))
        L = D - A
        eigvecs = self.eigvecs
        cluster_assignments = eigvecs[:, self.sorted_indices]
        predicted_labels = self.kmeans_model.predict(cluster_assignments)
        return predicted_labels
