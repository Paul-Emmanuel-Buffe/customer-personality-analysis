```markdown
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

```

---

### B. Confrontation au Standard Industriel (Benchmark)

Afin de valider la rigueur mathématique du code développé, un protocole de test comparatif a été mis en place. Pour simuler une phase d'exploration à l'aveugle, le jeu de données Iris a été traité sans prendre en compte ses 3 classes réelles, en imposant volontairement une sur-segmentation forte à 10 clusters ($k = 10$).

```python
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    print("\n=== BENCHMARK: K-MEANS MAISON VS SCIKIT-LEARN SUR IRIS ===")

    # 1. Chargement des données Iris
    iris = load_iris()
    X = iris.data

    # 2. CONFIGURATION DU NOMBRE DE CLUSTERS 
    CHOIX_K = 10
    np.random.seed(42) # reproducibilité des résultats pour le modèle maison

    # 3. Instanciation et entraînement
    model_maison = KMeansMaison(n_clusters=CHOIX_K)
    model_maison.fit(X)

    # 4. CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE SCIKIT-LEARN
    model_sklearn = KMeans(n_clusters=CHOIX_K, random_state=42, n_init=10, max_iter=100)
    model_sklearn.fit(X)

```

---

### C. Analyse Synthétique des Résultats à k = 10

L'exécution de ce premier benchmark à $k=10$ met en évidence l'impact des stratégies d'initialisation et introduit la nécessité d'une double évaluation métrique (Inertie + Silhouette).

```
--------------------------------------------------
 COMPARAISON DES PERFORMANCES (Pour k = 10)
-> Inertie K-Means Maison       : 28.29
-> Inertie K-Means Scikit-Learn : 25.97
-> Silhouette K-Means Maison    : 0.3078
-> Silhouette Scikit-Learn      : 0.3180
--------------------------------------------------

```

#### 1. Optimisation de l'Inertie et Rôle du Multi-Start

L'algorithme de Scikit-Learn atteint une inertie globale plus faible (25.97 contre 28.29), traduisant des clusters géométriquement plus compacts. Cet écart provient directement du paramètre `n_init=10`. Alors que le modèle personnalisé s'exécute en une seule tentative ("single-shot") soumise au hasard du tirage initial, Scikit-Learn effectue 10 lancements indépendants complets et ne conserve que la meilleure convergence, évitant ainsi le piège des minima locaux.

#### 2. Quantification du Surapprentissage par la Silhouette

L'introduction du score de Silhouette dévoile une **illusion mathématique** : courir après l'inertie la plus basse en augmentant le nombre de groupes fait s'effondrer la qualité du clustering à un score médiocre d'environ **0.31**. La métrique sanctionne la sur-segmentation en prouvant que les 10 clusters créés sont artificiels, trop proches et se chevauchent massivement dans l'espace.

#### 3. Divergence des Frontières de Décision

L'examen des 10 premiers échantillons confirme cette instabilité géométrique par une indexation et une répartition structurellement différentes :

* **Labels Maison :** `[5 5 5 5 5 1 5 5 5 5]`

* **Labels Scikit-Learn :** `[2 8 8 8 2 7 8 2 8 8]`


Au-delà de la simple permutation des numéros de groupes, les frontières de décision divergent (Scikit-Learn sépare la première et la deuxième fleur là où le modèle maison les maintient ensemble). Le modèle industriel affine la fragmentation grâce à des centroïdes initiaux mieux positionnés. Néanmoins, l'écart de Silhouette infime (**0.01**) prouve que la logique mathématique globale du modèle Maison est parfaitement fonctionnelle.

---

### D. Arbitrage de k : Entre Illusion Mathématique et Réalité Métier

Pour s'extraire du piège de la sur-segmentation, une étude approfondie des différents régimes de partitionnement ($k=2$, $k=3$ et $k=10$) a été menée. Elle met en lumière l'obligation d'arbitrer entre performance brute et pertinence métier.

#### 2. Analyse des Comportements Émergents

* **Sous-apprentissage ($k$ trop bas) :** À $k=2$, le score de Silhouette culmine à **0.6810** alors que l'inertie est mauvaise. C'est un paradoxe géométrique : le modèle fusionne deux espèces distinctes (Versicolor et Virginica) en un seul macro-groupe car elles sont très proches. La Silhouette est excellente uniquement parce que la distance avec la troisième classe isolée (Setosa) est immense, masquant ainsi la structure réelle des données.


* **Illusion d'inertie ($k$ trop haut) :** À $k=10$, l'inertie est excellente mais le score de Silhouette s'effondre. Le modèle sur-apprend en fragmentant artificiellement des populations biologiquement homogènes pour satisfaire un critère purement mathématique.


* **Méthode du coude ($k$ optimal) :** C'est ici que la **méthode du Coude (Elbow)** prend tout son sens. En cartographiant l'évolution de l'inertie, elle permet de détecter visuellement le point d'inflexion exact à **$k=3$**. Ce point représente l'équilibre parfait : l'algorithme cesse de segmenter dès que le gain de structure s'amoindrit, offrant un modèle hautement interprétable et en parfaite adéquation avec la réalité du terrain.



Pour valider cette intuition géométrique directement sur le modèle personnalisé, le protocole suivant a été appliqué :

```python
# 1. Calcul de l'inertie pour k allant de 1 à 10
inerties_maison = []
k_range = range(1, 11)

for k in k_range:
    np.random.seed(42) # Fixation de la graine pour stabiliser le tirage de chaque k
    model = KMeansMaison(n_clusters=k, max_iter=100)
    model.fit(X)
    inerties_maison.append(model.inertia)

# 2. Création de la visualisation "Elbow"
plt.figure(figsize=(9, 5))
plt.plot(k_range, inerties_maison, marker='o', linestyle='--', color='#ff7f0e', linewidth=2, markersize=8)
plt.axvline(x=3, color='#d62728', linestyle=':', linewidth=2, label='Coude optimal (k = 3)')

plt.title("Méthode du Coude (Elbow Method) - Modèle KMeans Maison", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Nombre de clusters (k)", fontsize=12)
plt.ylabel("Inertie intra-classe globale", fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig("customers_ressources/elbow.png") # Enregistrement automatique de l'image
plt.show()

```

## 5. Réduction de Dimension : L'Analyse à Facteurs Multiples (AFM)

Pour modéliser les comportements des clients de l'épicerie, une étape de réduction de dimension est nécessaire avant d'appliquer nos algorithmes de clustering. Cependant, une Analyse en Composantes Principales (ACP) classique se heurte à deux problèmes majeurs sur ce type de jeu de données :

* **L'hétérogénéité des types de données :** Le dataset contient à la fois des variables quantitatives (âge, revenus, montants dépensés) et qualitatives (niveau d'études, statut marital).


* **Le déséquilibre des groupes de variables :** Le bloc des dépenses compte 6 variables, tandis que le profil qualitatif n'en compte que 2. Une ACP classique donnerait mathématiquement trois fois plus de poids aux dépenses, écrasant les autres dimensions du profil client.



La solution pour pallier ces biais réside dans l'utilisation de l'**Analyse à Facteurs Multiples (AFM)** (ou *Multiple Factor Analysis - MFA*).

---

### A. L'Analogie du "Jury" (L'intuition mathématique)

L'AFM permet d'équilibrer l'influence de plusieurs blocs sémantiques. Imaginons que l'on demande à 4 "Jurys" distincts d'évaluer les clients :

1. **Les Sociologues** (Socio-Démo : Âge, Revenus, Enfants, Ancienneté).


2. **Les Comptables** (Dépenses : Vins, Viandes, Or, etc.).


3. **Les Techniciens** (Canaux d'achat : Web, Magasin, Catalogue).


4. **L'État Civil** (Qualitatif : Éducation, Statut marital).



Si tous parlent en même temps dans un algorithme classique, les Comptables feront plus de bruit simplement car ils possèdent plus de variables. L'AFM agit comme un modérateur intra-algorithmique : elle accorde **exactement le même volume sonore maximal** à chaque Jury, indépendamment de sa taille initiale.

---

### B. Le Mécanisme Mathématique de l'AFM

L'algorithme procède en trois étapes logiques pour standardiser et comparer ces blocs :

1. **L'analyse isolée :** L'AFM réalise d'abord une ACP (pour le quantitatif) ou une ACM (pour le texte/qualitatif) séparément sur chaque groupe de variables.


2. **La pondération stricte :** Pour chaque bloc, l'algorithme identifie son axe d'information le plus fort (sa première valeur propre, notée $\lambda_1$). Il divise ensuite toutes les variables de ce bloc par cette valeur. Résultat : chaque bloc a désormais une "force" maximale (inertie) strictly égale à 1.


3. **L'analyse globale :** Une fois les groupes mis sur un pied d'égalité, l'AFM fusionne tous les blocs pondérés et réalise une analyse globale pour extraire les grandes "super-dimensions" (composantes principales) qui structurent réellement la clientèle.



---

### C. Les Dimensions Factorielles et le Scree Plot

En sortie, l'AFM ne restitue pas les variables brutes. Elle génère de nouvelles variables synthétiques et non corrélées, appelées les **Dimensions** (ou composantes).

* **Dimension 1 :** L'axe expliquant la plus grande part de variance dans la base client (ex: il pourrait opposer les acheteurs "Premiums/Vins" aux "Chasseurs de Promos").


* **Dimension 2 :** Le deuxième axe le plus important, orthogonal (totalement indépendant) au premier.



#### Filtrage du Bruit (Scree Plot)

Afin de déterminer le nombre optimal de dimensions à conserver en entrée de nos algorithmes de clustering (K-Means, CAH, DBSCAN), nous utilisons un **Scree Plot** (graphique des valeurs propres). En observant le pourcentage de variance expliquée par chaque dimension, nous recherchons la cassure (le "coude") où l'apport d'information stagne. Cela permet de conserver un "signal pur" et robuste tout en rejetant le "bruit" statistique, optimisant ainsi la stabilité des clusters finaux.

```
