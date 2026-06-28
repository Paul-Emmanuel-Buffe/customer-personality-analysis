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
        CONSTRUCTEUR: Configurtion  du modèle K-Means.
        -k: Nombre de groupes (clusters) à former.
        -max_iter: Nombre maximum d'itérations pour l'algorithme.
        """
        self.k = n_clusters 
        self.max_iter = max_iter
        self.centroids = None     
        self.labels = None        
        self.inertia = None       
    
    def initialize_centroids(self, X): 
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


# =====================================================================
# BLOC EXÉCUTION 
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    print("\n=== BENCHMARK: K-MEANS MAISON VS SCIKIT-LEARN SUR IRIS ===")

    # 1. Chargement des données Iris
    iris = load_iris()
    X = iris.data

    # 2. CONFIGURATION DU NOMBRE DE CLUSTERS 
    CHOIX_K = 10
    np.random.seed(42) # reproducibilité des résultats pour le model maison

    # 3. Instanciation et entraînement
    model_maison = KMeansMaison(n_clusters=CHOIX_K)
    model_maison.fit(X)

    #4 CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE SCIKIT-LEARN
    model_sklearn = KMeans(n_clusters=CHOIX_K, random_state=42, n_init=10, max_iter=100)
    model_sklearn.fit(X)

    # 4. AFFICHAGE DU COMPARATIF DES RÉSULTATS
    print("\n--------------------------------------------------")
    print(f" COMPARAISON DE L'INERTIE (Pour k = {CHOIX_K})")
    print(f"-> Inertie K-Means Maison      : {model_maison.inertia:.2f}")
    print(f"-> Inertie K-Means Scikit-Learn : {model_sklearn.inertia_:.2f}")
    print("--------------------------------------------------")
    
    print("\n  COMPARAISON DES 10 PREMIERS LABELS")
    print(f"-> Labels Maison      : {model_maison.labels[:10].astype(int)}")
    print(f"-> Labels Scikit-Learn : {model_sklearn.labels_[:10]}")
    print("--------------------------------------------------")


# =====================================================================
    # VISUALISATION EN 3D Modèle Maison
    # =====================================================================
    import matplotlib.pyplot as plt

    names = iris.feature_names
    customcmap = "tab10"  


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Affichage des points selon les prédictions dumodèle
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], 
               c=model_maison.labels.astype(float), 
               edgecolor="k", s=150, cmap=customcmap)

    ax.view_init(20, -50)
    ax.set_xlabel(names[3], fontsize=12)
    ax.set_ylabel(names[0], fontsize=12)
    ax.set_zlabel(names[2], fontsize=12)
    ax.set_title(f"K-Means Clusters (Modèle Maison, k={CHOIX_K})", fontsize=14, fontweight='bold')

    plt.show()