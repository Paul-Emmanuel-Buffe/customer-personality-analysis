# Customer-personality-analysis

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


## Veille Technologique : Classification Non Supervisée

---
## 2. Analyse des 3 Algorithmes Étudiés

### 1. K-Means (Partitionnement)

* **Principe :** On fixe $k$ (le nombre de groupes). L'algorithme place des "points de ralliement" au hasard, puis chaque client rejoint le centre le plus proche. Le centre se déplace ensuite au milieu exact du nouveau groupe formé. On répète jusqu'à ce que les centres ne bougent plus.
* **Force :** Haute performance sur les grands jeux de données ; interprétabilité simple.
* **Limite :** $k$ doit être défini *a priori* ; vulnérable aux valeurs aberrantes (outliers).
* **Usage idéal :** Segmentation client.

#### A. Formule de la Distance (Distance Euclidienne)

$$dist(p, q) = \sqrt{\sum_{j=1}^{n} (p_j - q_j)^2}$$

* **$dist(p, q)$ :** La distance "à vol d'oiseau" entre deux points (le client $p$ et le centre $q$).
* **$\sqrt{\quad}$ :** La racine carrée, utilisée pour obtenir une distance réelle après avoir élevé les écarts au carré.
* **$\sum$ :** Le symbole de la somme, indiquant qu'on additionne les écarts de toutes les caractéristiques.
* **$j$ :** L'index d'une caractéristique spécifique (ex: l'âge, le revenu).
* **$n$ :** Le nombre total de caractéristiques (la dimensionnalité des données).
* **$(p_j - q_j)$ :** L'écart entre la valeur de la caractéristique $j$ pour le client et la valeur de cette même caractéristique pour le centre.
* **$^2$ :** L'exposant au carré, qui permet de rendre toutes les différences positives (les écarts négatifs deviennent positifs).

#### B. Formule de l'Inertie (Compacité du groupe)

$$W = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

* **$W$ :** L'inertie totale (la mesure de la dispersion des points dans leurs groupes).
* **$\sum_{i=1}^{k}$ :** La somme pour chaque groupe (de 1 jusqu'au nombre total de groupes $k$).
* **$\sum_{x \in C_i}$ :** La somme pour chaque point $x$ appartenant au groupe $C_i$.
* **$||x - \mu_i||^2$ :** La distance au carré entre le point $x$ et le centre du groupe $\mu_i$ (le centroïde).
* **$\mu_i$ :** La position du centre (moyenne) du groupe $i$.

---

### 2. DBSCAN (Basé sur la densité)

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

### 3. CAH (Classification Ascendante Hiérarchique)

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