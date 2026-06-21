import numpy as np

def euclidean_distance(x1, x2):
    """
    Calcul de la distance euclidienne entre deux points dans un espace n-dimensionnel.
    Parametres :
    x1 (array-like): Coordonnées du point1 .
    x2 (array-like): Coordonnées du point2 .
    Returns:
    nombre flottant de la distance euclidienne.
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KMeansMaison:
    def __init__(self, n_clusters=3, max_iter=100):
        """ 
        CONSTRUCTEUR: Configurtion initile du modèle K-Means.
        -k: Nombre de groupes (clusters) à former.
        -max_iter: Nombre maximum d'itérations pour l'algorithme.
        """
        self.k = n_clusters # CORRIGÉ : utilise n_clusters reçu en argument
        self.max_iter = max_iter
        self.centroids = None     
        self.labels = None        
        self.inertia = None       
    
    def initialize_centroids(self, X): # CORRIGÉ : orthographe de initialize
        """
        ETAPE 1: Le point de départ
        On choisit aléatoirement k points comme centroïdes initiaux.
        """
        centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
        return X[centroids_idx]  


    def assign_clusters(self, X, centroids):
        """ # CORRIGÉ : retrait du guillemet en trop (3 au lieu de 4)
        ETAPE 2: Le tri des données
        """
        clusters = [[] for _ in range(self.k)]

        for idx_point, point in enumerate(X):
            distances = [euclidean_distance(point, center) for center in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx_point)
        return clusters


    def update_centroids(self, X, clusters):
        """
        ÉTAPE 3 : Le déménagement des centres.
        """
        centroids = np.zeros((self.k, X.shape[1]))

        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                centroids[i] = self.centroids[i]  
            else:
                centroids[i] = X[cluster].mean(axis=0)

        return centroids

    def fit(self, X):
        """
        Orcherstration de l'algorithme K-Means La boucle d'entrainement
        """
        # Étape 1 : Initialisation des centroïdes
        self.centroids = self.initialize_centroids(X) # CORRIGÉ : orthographe de initialize

        for _ in range(self.max_iter):
            # Étape 2 : Assignation des échantillons aux clusters
            clusters = self.assign_clusters(X, self.centroids)

            prev_centroids = self.centroids
            
            # Les centres se déplacent vers la moyenne de leur groupe
            self.centroids = self.update_centroids(X, clusters)

            # Condition d'arrêt
            if np.allclose(self.centroids, prev_centroids):
                break

        # Attribution des labels
        self.labels = np.zeros(X.shape[0])
        for cluster_idx, point_indices in enumerate(clusters):
            self.labels[point_indices] = cluster_idx

        # Calcul de l'inertie
        self.inertia = 0
        for cluster_idx, point_indices in enumerate(clusters):
            for idx in point_indices:
                self.inertia += euclidean_distance(X[idx], self.centroids[int(self.labels[idx])]) ** 2