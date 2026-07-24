# Projet : Reconnaissance d'Objets du Quotidien en Temps Réel

### Installation et Prérequis

Pour exécuter ce projet, vous devez installer plusieurs bibliothèques Python. Il est fortement recommandé de créer un environnement virtuel au préalable.

Vous pouvez installer toutes les dépendances en une seule commande :

```bash
pip install tensorflow opencv-python scikit-learn tensorboard seaborn matplotlib pandas numpy pillow
```

## Ordre d'exécution des scripts

Pour reproduire ce projet et entraîner le modèle, veuillez exécuter les scripts dans l'ordre strict suivant :

### 1. Collecte des images (si vous n'avez pas encore le dataset)

```bash
python src/capture_data.py
```

**Action :** Ouvre la webcam pour capturer les images des objets. Les images sont sauvegardées dans le dossier `dataset/`.

### 2. Préparation et séparation des données

```bash
python src/preprocess_data.py
```

**Action :** Divise aléatoirement les images collectées en un jeu d'entraînement (80 %) et un jeu de test (20 %). Crée le dossier `dataset/processed/`.

### 3. Entraînement du modèle

```bash
python src/training.py
```

**Action :** Charge les images, normalise les pixels, entraîne le modèle de Deep Learning et génère les logs TensorBoard. Le meilleur modèle est sauvegardé dans `models/best_model.keras`.

### 4. Évaluation du modèle

```bash
python src/evaluate.py
```