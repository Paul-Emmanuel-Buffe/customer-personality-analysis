# Customer Personality Analysis

![Clusters Space Header](customers_ressources/clusters_space.png)

---

## 1. Introduction à la Classification Non Supervisée

Dans un marché saturé, la segmentation démographique classique (âge, genre) ne suffit plus. Ce projet de **Customer Personality Analysis** utilise le **Machine Learning non supervisé** (clustering) pour regrouper dynamiquement les clients selon leurs comportements d'achat réels. 

La classification non supervisée (ou *clustering*) est une branche du Machine Learning où l'algorithme doit apprendre à structurer des données sans aucune étiquette (label) préalable. L'objectif est de regrouper les individus au sein de clusters homogènes, de telle sorte que :
- Les individus d'un même groupe soient les plus **similaires** possibles (cohésion).
- Les différents groupes soient les plus **distants** possibles (séparation).

---

### La Problématique

> *Comment structurer une base clients hétérogène en groupes homogènes, sans étiquettes préalables, pour personnaliser les stratégies marketing et maximiser le ROI ?*

---

### Objectifs du Projet

* **Veille Technologique :** Évaluer et comparer 3 algorithmes clés (**K-Means, DBSCAN, CAH**).
* **Préparation des Données (EDA) :** Nettoyer et normaliser le dataset (Revenus, Score de dépenses).
* **Modélisation & Optimisation :** Ajuster les hyperparamètres et valider la qualité des clusters via le **score de Silhouette**.
* **Déploiement Métier :** Traduire les groupes mathématiques en *personas* et actions marketing concrètes.

---

## 2. Veille Technologique : Analyse des 3 Algorithmes Étudiés

### A. K-Means (Partitionnement)

* **Principe :** On fixe $k$ (le nombre de groupes). L'algorithme place des "points de ralliement" au hasard, puis chaque client rejoint le centre le plus proche. Le centre se déplace ensuite au milieu exact du nouveau groupe formé. On répète jusqu'à ce que les centres ne bougent plus.
* **Force :** Haute performance sur les grands jeux de données ; interprétabilité simple.
* **Limite :** $k$ doit être défini *a priori* ; vulnérable aux valeurs aberrantes (outliers).
* **Usage idéal :** Segmentation client.

#### Formule de la Distance (Distance Euclidienne)

$$dist(p, q) = \sqrt{\sum_{j=1}^{n} (p_j - q_j)^2}$$

* **$dist(p, q)$ :** La distance "à vol d'oiseau" entre deux points (le client $p$ et le centre $q$).
* **$\sqrt{\quad}$ :** La racine carrée, utilisée pour obtenir une distance réelle après avoir élevé les écarts au carré.
* **$\sum$ :** Le symbole de la somme, indiquant qu'on additionne les écarts de toutes les caractéristiques.
* **$j$ :** L'index d'une caractéristique spécifique (ex: l'âge, le revenu).
* **$n$ :** Le nombre total de caractéristiques (la dimensionnalité des données).
* **$(p_j - q_j)$ :** L'écart entre la valeur de la caractéristique $j$ pour le client et la valeur de cette même caractéristique pour le centre.
* **$^2$ :** L'exposant au carré, qui permet de rendre toutes les différences positives (les écarts négatifs deviennent positifs).

#### Formule de l'Inertie (Compacité du groupe)

$$W = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

* **$W$ :** L'inertie totale (la mesure de la dispersion des points dans leurs groupes).
* **$\sum_{i=1}^{k}$ :** La somme pour chaque groupe (de 1 jusqu'au nombre total de groupes $k$).
* **$\sum_{x \in C_i}$ :** La somme pour chaque point $x$ appartenant au groupe $C_i$.
* **$||x - \mu_i||^2$ :** La distance au carré entre le point $x$ et le centre du groupe $\mu_i$ (le centroïde).
* **$\mu_i$ :** La position du centre (moyenne) du groupe $i$.

---

### B. DBSCAN (Basé sur la densité)

* **Principe :** Regroupe les points selon leur densité locale. Les zones clairsemées sont marquées comme "bruit".
* **Force :** Aucun $k$ à définir ; capable d'identifier des formes non-sphériques complexes.
* **Limite :** Très sensible au paramètre de voisinage ($\epsilon$).
* **Usage idéal :** Détection d'anomalies (outliers) et données spatiales complexes.

#### Formule du Voisinage (Critère de densité)

$$N_\epsilon(p) = \{q \in D \mid dist(p, q) \le \epsilon\}$$

* **$N_\epsilon(p)$ :** Le voisinage du point $p$ (l'ensemble des points voisins).
* **$\{q \in D\}$ :** L'ensemble de tous les points $q$ disponibles dans ton jeu de données $D$.
* **$dist(p, q)$ :** La distance (euclidienne) entre le point $p$ et le point $q$.
* **$\le \epsilon$ :** La condition qui définit la limite du voisinage (le rayon epsilon).

---

### C. CAH (Classification Ascendante Hiérarchique)

* **Principe :** Approche "bottom-up". Chaque point débute comme un cluster unique. À chaque itération, les deux clusters les plus proches fusionnent.
* **Force :** Pas besoin de définir $k$ à l'avance ; dendrogramme pour visualiser la hiérarchie.
* **Limite :** Complexité calculatoire élevée ($O(n^3)$), inadapté aux très gros volumes de données.
* **Usage idéal :** Analyse exploratoire où la compréhension de la structure est prioritaire.

#### Formule de la Distance de Ward (Critère de fusion)

$$\Delta(C_i, C_j) = \frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} ||\mu_i - \mu_j||^2$$

* **$\Delta(C_i, C_j)$ :** L'augmentation de l'inertie totale causée par la fusion des groupes $C_i$ et $C_j$.
* **$|C_i|$ et $|C_j|$ :** Le nombre de points contenus dans chaque groupe.
* **$|C_i| + |C_j|$ :** Le nombre total de points après la fusion.
* **$||\mu_i - \mu_j||^2$ :** La distance au carré entre les deux centres (centroïdes) des groupes.

> **Objectif de la formule :** Elle minimise l'étalement du nouveau groupe en favorisant la fusion de groupes déjà compacts et de taille similaire.

---

## 3. Sélection du Nombre Optimal de Clusters et Mesure de Qualité

Dans un contexte d'apprentissage non supervisé, les données ne possèdent pas d'étiquettes de vérité terrain (*ground truth*). L'analyste doit donc s'appuyer sur des critères mathématiques internes pour évaluer la pertinence du découpage et choisir la configuration idéale.

---

### A. La Méthode du Coude (Elbow Method)

Cette méthode est principalement utilisée pour déterminer la valeur optimale du nombre de clusters $k$ dans les algorithmes de partitionnement comme le K-Means.

#### 1. Principe
L'algorithme K-Means est exécuté plusieurs fois en faisant varier $k$ (par exemple de 1 à 10). Pour chaque valeur de $k$, on calcule l'**Inertie intra-classe totale** ($W$). On trace ensuite la courbe de l'inertie en fonction de $k$.
L'inertie diminue naturellement à chaque fois que l'on ajoute un cluster. On recherche visuellement le point d'inflexion de la courbe (le "coude") : c'est le point où l'ajout d'un cluster supplémentaire n'apporte plus de gain significatif en termes de compacité.


#### 2. Rappel de la formule de l'Inertie ($W$)

$$W = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

* **$W$ :** L'inertie intra-classe globale. Plus elle est proche de 0, plus les clusters sont compacts.
* **$k$ :** Le nombre de clusters testé.
* **$C_i$ :** Le cluster numéro $i$.
* **$x$ :** Un point de données affecté au cluster $C_i$.
* **$\mu_i$ :** Le centroïde (centre géométrique) du cluster $C_i$.
* **$||x - \mu_i||^2$ :** La distance euclidienne au carré entre le point et son centre.

![Clusters Space Header](customers_ressources/elbow_fonctionnement.png)
---

### B. Le Score de Silhouette (Silhouette Coefficient)

Le score de Silhouette est une mesure globale et individuelle de la qualité du clustering. Il évalue si un point est correctement positionné au sein de son groupe par rapport aux groupes voisins.

#### 1. Formule du coefficient pour un individu $i$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

#### 2. Explication détaillée des symboles

* **$s(i)$ :** Le score de silhouette de l'observation $i$. Ce score est toujours compris entre -1 et 1.
* **$a(i)$ (La Cohésion) :** La distance moyenne entre l'observation $i$ et tous les autres points appartenant au **même** cluster. Plus $a(i)$ est petit, plus l'observation est proche de ses pairs (groupe compact).
* **$b(i)$ (La Séparation) :** La distance moyenne entre l'observation $i$ et tous les points du cluster **voisin le plus proche** (le cluster dont la distance moyenne avec $i$ est minimale, hors de son propre groupe). Plus $b(i)$ est grand, plus l'observation est isolée des autres groupes.
* **$\max(a(i), b(i))$ :** Le facteur de normalisation. Il prend la valeur la plus grande entre $a(i)$ et $b(i)$ pour s'assurer que le résultat final reste strictement confiné dans l'intervalle $[-1, 1]$.

#### 3. Interprétation du Score Global

Le score global du modèle est la moyenne des scores $s(i)$ de toutes les observations.

* **Proche de 1 :** Les clusters sont denses, bien séparés et chaque point est à sa place.
* **Proche de 0 :** Les clusters se chevauchent de manière importante ; les points sont situés sur les frontières de décision.
* **Proche de -1 :** Les points ont été affectés au mauvais cluster (ils sont plus proches du groupe voisin que du leur).

![Clusters Space Header](customers_ressources/score_silhouette.png)
