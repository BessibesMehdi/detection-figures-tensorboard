# Détection et Classification de Figures Géométriques (CNN)

Projet de détection d'objets et de classification de formes géométriques (*cercles, carrés, triangles*) de couleurs variées dans des images RGB synthétiques 64×64. Ce projet intègre la visualisation en direct des métriques sous **TensorBoard**.

Projet réalisé dans le cadre du TP et mini-projet de Réseaux de Neurones (**Extension Option B : Plusieurs classes de formes**).

---

## 📌 Fonctionnalités & Architecture

### 1. Données d'entrée & Générateur Multi-formes (Option B)
- Images synthétiques RGB de dimension $64 \times 64$.
- 3 classes de formes géométriques :
  - `0` : Cercle (`circle`)
  - `1` : Carré (`square`)
  - `2` : Triangle (`triangle`)
- Couleurs variées (rouge, bleu, vert, jaune, violet) et positions/tailles aléatoires.

### 2. Architecture du Détecteur (`Detector`)
- **Backbone** : 3 blocs convolutionnels avec stride 2, BatchNorm2d et activation ReLU :
  - Conv2d(3 $\to$ 8) + BatchNorm2d + ReLU
  - Conv2d(8 $\to$ 16) + BatchNorm2d + ReLU
  - Conv2d(16 $\to$ 32) + BatchNorm2d + ReLU
  - Réduit la résolution spatiale de $64 \times 64$ à une grille de feature maps de dimension $8 \times 8$ (stride = 8) sur 32 canaux.
- **Head (Tête de prédiction)** : Convolution $1 \times 1$ passant de 32 canaux à **8 canaux de sortie** :
  - Canal 0 : Score d'objectness $p_{\text{obj}}$ (présence d'un objet dans la cellule).
  - Canaux 1 à 4 : Paramètres de la boîte englobante $(\Delta x, \Delta y, w, h)$.
  - Canaux 5 à 7 : Logits de classification (cercle, carré, triangle).

### 3. Fonction de Perte Multi-tâche (`compute_loss`)
La fonction de perte totale combine trois objectifs :
$$\mathcal{L} = \mathcal{L}_{\text{objectness}} + 10 \cdot \mathcal{L}_{\text{boîte}} + \mathcal{L}_{\text{classe}}$$
- $\mathcal{L}_{\text{objectness}}$ : `BCEWithLogitsLoss` calculée sur toutes les cellules de la grille avec pondération positive `pos_weight = 10.0`.
- $\mathcal{L}_{\text{boîte}}$ : `MSELoss` calculée sur les coordonnées normalisées de la boîte de la cellule positive (contenant le centre de l'objet).
- $\mathcal{L}_{\text{classe}}$ : `CrossEntropyLoss` calculée sur les logits des 3 formes pour la cellule positive.

### 4. Post-traitement & Évaluation
- **Décodage & NMS** : Conversion des prédictions vers l'espace image et Non-Maximum Suppression pour ne retenir que la boîte de confiance maximale.
- **Métriques d'évaluation** :
  - **Localisation** : IoU moyen et proportion d'IoU $> 0.5$.
  - **Classification** : Exactitude de classification (`Accuracy_class`).
  - Suivi en temps réel de toutes les métriques dans **TensorBoard**.

---

## 📁 Structure du Projet

```text
├── notebooks/
│   └── demo.ipynb             # Notebook complet : dataset, entraînement, TensorBoard et visualisation
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py   # Générateur de base (disque unique)
│   ├── dataset_generator_2.py # Générateur Option B (cercles, carrés, triangles, couleurs)
│   ├── detector.py            # Modèle CNN, encodage/décodage, loss multi-tâche, predict
│   ├── train.py               # Boucle d'entraînement et fonction d'évaluation (IoU + Class Accuracy)
│   └── visualize.py           # Fonctions d'affichage des images, feature maps et détections
├── requirements.txt           # Dépendances du projet
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Utilisation

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Entraînement et démo
Ouvrez et exécutez le notebook Jupyter :
```bash
jupyter notebook notebooks/demo.ipynb
```

### 3. Visualisation avec TensorBoard
Pour lancer l'interface TensorBoard et observer les courbes d'apprentissage et de validation :
```bash
tensorboard --logdir="C:/Users/<utilisateur>/AppData/Local/Temp/tb_runs/demo" --port=6006
```
Puis accédez à [http://localhost:6006](http://localhost:6006) dans votre navigateur.
