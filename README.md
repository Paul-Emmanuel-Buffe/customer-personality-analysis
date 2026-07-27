# Customer Personality Analysis

![Clusters Space Header](customers_ressources/clusters_space.png)

---

## 1. Introduction à la Classification Non Supervisée

Dans un marché saturé, la segmentation démographique classique (âge, genre) ne suffit plus. Ce projet de **Customer Personality Analysis** utilise le **Machine Learning non supervisé** (clustering) pour regrouper dynamiquement les clients selon leurs comportements d'achat réels[cite: 1, 2]. 

La classification non supervisée (ou *clustering*) est une branche du Machine Learning où l'algorithme doit apprendre à structurer des données sans aucune étiquette (label) préalable[cite: 1, 2]. L'objectif est de regrouper les individus au sein de clusters homogènes, de telle sorte que :
- Les individus d'un même groupe soient les plus **similaires** possibles (cohésion)[cite: 2].
- Les différents groupes soient les plus **distants** possibles (séparation)[cite: 2].

---

### La Problématique

> *Comment structurer une base clients hétérogène en groupes homogènes, sans étiquettes préalables, pour personnaliser les stratégies marketing et maximiser le ROI ?*[cite: 1, 2]

---

### Objectifs du Projet

* **Veille Technologique :** Évaluer et comparer 3 algorithmes clés (**K-Means, DBSCAN, CAH**)[cite: 1, 2].
* **Préparation des Données (EDA) :** Nettoyer et normaliser le dataset (Revenus, Score de dépenses)[cite: 1, 2].
* **Modélisation & Optimisation :** Ajuster les hyperparamètres et valider la qualité des clusters via le **score de Silhouette**[cite: 1, 2].
* **Déploiement Métier :** Traduire les groupes mathématiques en *personas* et actions marketing concrètes[cite: 1, 2].

---

## 2. Veille Technologique : Analyse des 3 Algorithmes Étudiés

### A. K-Means (Partitionnement)

* **Principe :** On fixe $k$ (le nombre de groupes). L'algorithme place des "points de ralliement" au hasard, puis chaque client rejoint le centre le plus proche. Le centre se déplace ensuite au milieu exact du nouveau groupe formé. On répète jusqu'à ce que les centres ne bougent plus[cite: 1, 2].
* **Force :** Haute performance sur les grands jeux de données ; interprétabilité simple[cite: 2].
* **Limite :** $k$ doit être défini *a priori* ; vulnérable aux valeurs aberrantes (outliers)[cite: 2].
* **Usage idéal :** Segmentation client[cite: 2].

#### Formule de la Distance (Distance Euclidienne)

$$dist(p, q) = \sqrt{\sum_{j=1}^{n} (p_j - q_j)^2}$$

* **$dist(p, q)$ :** La distance "à vol d'oiseau" entre deux points (le client $p$ et le centre $q$)[cite: 2].
* **$\sqrt{\quad}$ :** La racine carrée, utilisée pour obtenir une distance réelle après avoir élevé les écarts au carré[cite: 2].
* **$\sum$ :** Le symbole de la somme, indiquant qu'on additionne les écarts de toutes les caractéristiques[cite: 2].
* **$j$ :** L'index d'une caractéristique spécifique (ex: l'âge, le revenu)[cite: 2].
* **$n$ :** Le nombre total de caractéristiques (la dimensionnalité des données)[cite: 2].
* **$(p_j - q_j)$ :** L'écart entre la valeur de la caractéristique $j$ pour le client et la valeur de cette même caractéristique pour le centre[cite: 2].
* **$^2$ :** L'exposant au carré, qui permet de rendre toutes les différences positives (les écarts négatifs deviennent positifs)[cite: 2].

#### Formule de l'Inertie (Compacité du groupe)

$$W = \sum_{i=1}^{k} \sum_{x \in C_i} \vert{}\vert{}x - \mu_i\vert{}\vert{}^2$$

* **$W$ :** L'inertie totale (la mesure de la dispersion des points dans leurs groupes)[cite: 2].
* **$\sum_{i=1}^{k}$ :** La somme pour chaque groupe (de 1 jusqu'au nombre total de groupes $k$)[cite: 2].
* **$\sum_{x \in C_i}$ :** La somme pour chaque point $x$ appartenant au groupe $C_i$[cite: 2].
* **$\vert{}\vert{}x - \mu_i\vert{}\vert{}^2$ :** La distance au carré entre le point $x$ et le centre du groupe $\mu_i$ (le centroïde)[cite: 2].
* **$\mu_i$ :** La position du centre (moyenne) du groupe $i$[cite: 2].

---

### B. DBSCAN (Basé sur la densité)

* **Principe :** DBSCAN fonctionne par **voisinage et densité**. Un cluster est simplement défini comme une zone où la concentration de points est plus élevée que dans le reste de l'espace[cite: 2].
* **Absence de $k$ préimposé :** Vous n'avez pas à deviner ni à imposer le nombre de clusters au départ[cite: 2]. L'algorithme découvre de lui-même la structure naturelle des données : s'il y a 2 zones denses, il trouve 2 clusters ; s'il y en a 8, il en trouve 8[cite: 2].
* **Formes complexes :** Là où K-Means ne sait créer que des "bulles" (sphères), DBSCAN peut suivre des densités contiguës et découvrir des clusters en forme de croissants, d'anneaux, de serpents ou de structures complexes irrépartissables par des centres[cite: 2].
* **Apport clé (Gestion du bruit) :** Il sait dire *"ce point est trop isolé, il n'appartient à rien"*, ce qui en fait un outil parfait pour isoler les anomalies[cite: 2].
* **Usage idéal :** Détection d'anomalies (outliers) et données spatiales/densités complexes[cite: 2].

#### Typologie des 3 Catégories de Points
DBSCAN classe chaque donnée de votre jeu de données dans l'une des 3 catégories suivantes :
* **Point Cœur (*Core point*) :** Un point qui possède au moins `min_samples` voisins dans son rayon `eps`. C'est le « moteur » d'un cluster[cite: 2].
* **Point Frontière (*Border point*) :** Un point qui n'a pas assez de voisins pour être un point cœur, mais qui se trouve dans le rayon `eps` d'un point cœur. Il forme la bordure du groupe[cite: 2].
* **Bruit / Outlier (*Noise*) :** Un point qui n'est ni un point cœur, ni un point frontière. Il est étiqueté `-1`[cite: 2].

#### Les Hyperparamètres Clés
1. **`eps` (Epsilon) :** Le rayon de recherche autour d'un point[cite: 2].
2. **`min_samples` :** Le nombre minimum d'individus requis pour considérer une zone comme "dense"[cite: 2].
* **L'apprentissage méthodique :** On ne choisit pas `eps` au hasard, mais en traçant la courbe de la $k$-distance pour repérer le point d'inflexion (le "coude")[cite: 2].

#### Formule du Voisinage (Critère de densité)

$$N_\epsilon(p) = \{q \in D \mid dist(p, q) \le \epsilon\}$$

* **$N_\epsilon(p)$ :** Le voisinage du point $p$ (l'ensemble des points voisins)[cite: 2].
* **$\{q \in D\}$ :** L'ensemble de tous les points $q$ disponibles dans ton jeu de données $D$[cite: 2].
* **$dist(p, q)$ :** La distance (euclidienne) entre le point $p$ et le point $q$[cite: 2].
* **$\le \epsilon$ :** La condition qui définit la limite du voisinage (le rayon epsilon)[cite: 2].

---

### C. CAH (Classification Ascendante Hiérarchique)

* **Principe :** Approche "bottom-up". Chaque point débute comme un cluster unique. À chaque itération, les deux clusters les plus proches fusionnent[cite: 2].
* **Force :** Pas besoin de définir $k$ à l'avance ; dendrogramme pour visualiser la hiérarchie[cite: 2].
* **Limite :** Complexité calculatoire élevée ($O(n^3)$), inadapté aux très gros volumes de données[cite: 2].
* **Usage idéal :** Analyse exploratoire où la compréhension de la structure est prioritaire[cite: 2].

#### Formule de la Distance de Ward (Critère de fusion)

$$\Delta(C_i, C_j) = \frac{\vert{}C_i\vert{} \cdot \vert{}C_j\vert{}}{\vert{}C_i\vert{} + \vert{}C_j\vert{}} \vert{}\vert{}\mu_i - \mu_j\vert{}\vert{}^2$$

* **$\Delta(C_i, C_j)$ :** L'augmentation de l'inertie totale causée par la fusion des groupes $C_i$ et $C_j$[cite: 2].
* **$\vert{}C_i\vert{}$ et $\vert{}C_j\vert{}$ :** Le nombre de points contenus dans chaque groupe[cite: 2].
* **$\vert{}C_i\vert{} + \vert{}C_j\vert{}$ :** Le nombre total de points après la fusion[cite: 2].
* **$\vert{}\vert{}\mu_i - \mu_j\vert{}\vert{}^2$ :** La distance au carré entre les deux centres (centroïdes) des groupes[cite: 2].

> **Objectif de la formule :** Elle minimise l'étalement du nouveau groupe en favorisant la fusion de groupes déjà compacts et de taille similaire[cite: 2].

---

## 3. Sélection du Nombre Optimal de Clusters et Mesure de Qualité

Dans un contexte d'apprentissage non supervisé, les données ne possèdent pas d'étiquettes de vérité terrain (*ground truth*)[cite: 2]. L'analyste doit donc s'appuyer sur des critères mathématiques internes pour évaluer la pertinence du découpage et choisir la configuration idéale[cite: 2].

---

### A. La Méthode du Coude (Elbow Method)

Cette méthode est principalement utilisée pour déterminer la valeur optimale du nombre de clusters $k$ dans les algorithmes de partitionnement comme le K-Means[cite: 2].

#### 1. Principe
L'algorithme K-Means est exécuté plusieurs fois en faisant varier $k$ (par exemple de 1 à 10)[cite: 2]. Pour chaque valeur de $k$, on calcule l'**Inertie intra-classe totale** ($W$)[cite: 2]. On trace ensuite la courbe de l'inertie en fonction de $k$[cite: 2].
L'inertie diminue naturellement à chaque fois que l'on ajoute un cluster[cite: 2]. On recherche visuellement le point d'inflexion de la courbe (le "coude") : c'est le point où l'ajout d'un cluster supplémentaire n'apporte plus de gain significatif en termes de compacité[cite: 2].

#### 2. Rappel de la formule de l'Inertie ($W$)

$$W = \sum_{i=1}^{k} \sum_{x \in C_i} \vert{}\vert{}x - \mu_i\vert{}\vert{}^2$$

* **$W$ :** L'inertie intra-classe globale. Plus elle est proche de 0, plus les clusters sont compacts[cite: 2].
* **$k$ :** Le nombre de clusters testé[cite: 2].
* **$C_i$ :** Le cluster numéro $i$[cite: 2].
* **$x$ :** Un point de données affecté au cluster $C_i$[cite: 2].
* **$\mu_i$ :** Le centroïde (centre géométrique) du cluster $C_i$[cite: 2].
* **$\vert{}\vert{}x - \mu_i\vert{}\vert{}^2$ :** La distance euclidienne au carré entre le point et son centre[cite: 2].

![Clusters Space Header](customers_ressources/elbow_fonctionnement.png)

---

### B. Le Score de Silhouette (Silhouette Coefficient)

Le score de Silhouette est une mesure globale et individuelle de la qualité du clustering[cite: 2]. Il évalue si un point est correctement positionné au sein de son groupe par rapport aux groupes voisins[cite: 2].

#### 1. Formule du coefficient pour un individu $i$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

#### 2. Explication détaillée des symboles

* **$s(i)$ :** Le score de silhouette de l'observation $i$. Ce score est toujours compris entre -1 et 1[cite: 2].
* **$a(i)$ (La Cohésion) :** La distance moyenne entre l'observation $i$ et tous les autres points appartenant au **même** cluster[cite: 2]. Plus $a(i)$ est petit, plus l'observation est proche de ses pairs (groupe compact)[cite: 2].
* **$b(i)$ (La Séparation) :** La distance moyenne entre l'observation $i$ et tous les points du cluster **voisin le plus proche** (le cluster dont la distance moyenne avec $i$ est minimale, hors de son propre groupe)[cite: 2]. Plus $b(i)$ est grand, plus l'observation est isolée des autres groupes[cite: 2].
* **$\max(a(i), b(i))$ :** Le facteur de normalisation[cite: 2]. Il prend la valeur la plus grande entre $a(i)$ et $b(i)$ pour s'assurer que le résultat final reste strictly confiné dans l'intervalle $[-1, 1]$[cite: 2].

#### 3. Interprétation du Score Global

Le score global du modèle est la moyenne des scores $s(i)$ de toutes les observations[cite: 2].

* **Proche de 1 :** Les clusters sont denses, bien séparés et chaque point est à sa place[cite: 2].
* **Proche de 0 :** Les clusters se chevauchent de manière importante ; les points sont situés sur les frontières de décision[cite: 2].
* **Proche de -1 :** Les points ont été affectés au mauvais cluster (ils sont plus proches du groupe voisin que du leur)[cite: 2].

![Clusters Space Header](customers_ressources/score_silhouette.png)

---

### C. La Méthode de la k-Distance (Calibrage de DBSCAN) et Comparaison

Contrairement à K-Means, DBSCAN ne cherche pas un nombre de groupes $k$, mais un rayon de densité **`eps`**[cite: 2]. Pour le déterminer scientifiquement, on utilise la courbe de la **$k$-distance**[cite: 2].

#### 1. Principe
On fixe $k = \text{min\_samples}$ (ex: 5). Pour chaque point du jeu de données, on calcule la distance jusqu'à son 5ᵉ plus proche voisin. On trie ensuite l'ensemble de ces distances par ordre croissant pour obtenir une courbe.

#### 2. Différence entre la Méthode du Coude (K-Means) et la $k$-Distance (DBSCAN)

| Méthode | Algorithme cible | Axe X | Axe Y | Objectif du "Coude" |
| :--- | :--- | :--- | :--- | :--- |
| **Méthode du Coude** | **K-Means** | Nombre de clusters ($k$) | Inertie intra-classe globale ($W$) | Trouver le $k$ optimal où la compacité n'augmente plus significativement[cite: 2]. |
| **Méthode de la $k$-Distance** | **DBSCAN** | Points triés par distance | Distance au $k$-ème voisin (`eps`) | Trouver la valeur de **`eps`** marquant le passage des zones denses au bruit[cite: 2]. |

* **Lecture de la $k$-Distance :** La partie plate représente le cœur dense de la population (faibles distances entre voisins). La cassure soudaine (le coude) représente la valeur exacte de **`eps`** à retenir : au-delà de ce seuil, les distances explosent vers la verticale, caractérisant les points isolés (*bruit / outliers*)[cite: 2].

---

## 4. Implémentation Personnalisée et Benchmark

### A. Algorithme K-Means "Maison"

Pour maîtriser les mécanismes géométriques du partitionnement, l'algorithme K-Means a été entièrement réimplémenté en Python à l'aide de la bibliothèque NumPy[cite: 1, 2]. 

```python
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
        CONSTRUCTEUR: Configuration du modèle K-Means.
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
        """ 
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
        Orchestration de l'algorithme K-Means : La boucle d'entraînement
        """
        # Étape 1 : Initialisation des centroïdes
        self.centroids = self.initialize_centroids(X) 

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
