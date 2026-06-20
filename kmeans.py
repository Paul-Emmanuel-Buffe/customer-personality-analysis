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
        self.k = k 
        self.max_iter = max_iter
        self.centroids = None     # Stockera les coordonnées des centroïdes de chaque groupe après l'ajustement du modèle
        self.labels = None        # Stockera les labels des groupes pour chaque échantillon après l'ajustement du modèle
        self.inertia = None       # Stockera la somme des distances au carré entre les échantillons et leurs centroïdes respectifs après l'ajustement du modèle
    
    def inttialize_centroids(self, X):
        """
        ETAPE 1: Le point de départ
        On choisit aléatoirement k points comme centroïdes initiaux.
        Ces points serviront de référence pour former les groupes (clusters) dans les étapes suivantes.

        """
        # np.random.choice selectionne aléatoirement des indices uniques (replace=False) dans l'intervalle [0, X.shape[0]) pour choisir les centroïdes initiaux.
        centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
        return X[centroids_idx]  # Retourne les coordonnées des centroïdes initiaux choisis aléatoirement.


    def assign_clusters(self, X, centroids):

        """"
        ETAPE 2: Le tri des données
        Chaque individu (échantillon) examine les 'k' centroides disponibles, calcule sa distance avec chacun d'eux, et choisit le centroïde le plus proche.
        
        """
        # On crée une liste de listes vides pour stocker les indices des échantillons assignés à chaque cluster.
        clusters = [[] for _ in range(self.k)]

        #On passe en revue chaque échantillon du dataset X.
        for idx, sample in enumerate(X):
            # On calcule la distance entre l'échantillon actuel et chaque centroïde.
            distances = [euclidean_distance(point, center) for center in centroids]
            # np.argmin trouve la position (0, 1, 2, ...) du centroïde le plus proche (celui avec la distance minimale).
            cluster_idx = np.argmin(distances)
            # On ajoute l'indice de l'échantillon actuel à la liste du cluster correspondant.
            clusters[cluster_idx].append(idx)
        return clusters
