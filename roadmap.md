# ROADMAP PROJET 

---

# PHASE 1 : Préparation & Architecture du projet

## Étape 1.1 : Choix des classes et objets

- Sélection de **4 objets du quotidien** visuellement distincts.
  - Exemple :
    - Badge portail
    - Stylo
    - Téléphone
- Ajout d'une **5ᵉ classe obligatoire : Background**
  - Fond vide
  - Bureau
  - Mains vides
  - Objets aléatoires

---
## Étape 1.2 : Architecture du projet

```text
data/
│
├── raw/
│   └── Captures webcam brutes
│
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
src/
│
├── capture_photos.py
├── preprocess_data.py
├── model_builder.py
├── train_phase3.py
├── evaluate.py
├── train_phase4.py
└── inference_realtime.py
│
models/
│
├── baseline_model.keras
└── best_model.keras
│
logs/
└── TensorBoard
```

---

## Étape 1.3 : Environnement Python

Créer un environnement virtuel :

- `conda`

Installer les bibliothèques :

```text
tensorflow
keras
opencv-python
numpy
matplotlib
pandas
scikit-learn
tensorboard
```

---

# PHASE 2 : Constitution & Préparation du Dataset

## Étape 2.1 : Acquisition des images (`capture_photos.py`)

Objectifs :

- Capture webcam avec OpenCV
- Déclenchement via la touche **Espace**
- Nom des fichiers avec timestamp
- Entre **150 et 250 images** par objet
- Plus de **150 images** pour Background

Total attendu :

> Environ **1000 images**

Les prises doivent varier :

- angle
- distance
- luminosité
- orientation

---

## Étape 2.2 : Prétraitement (`preprocess_data.py`)

### Nettoyage

- Suppression des images floues
- Suppression des images corrompues

Vérification de la répartition :

```python
np.unique(labels, return_counts=True)
```

---

### Redimensionnement


- **128×128×3** (CNN personnalisé)

---

### Normalisation

Pixels :

```python
image = image / 255.0
```

Valeurs finales :

```
0 → 1
```

---

### Conversion du type

```python
X = X.astype("float32")
y = y.astype("float32")
```

---

### Découpage du dataset

Répartition classique :

- 80 % Train
- 20 % Test

---

# PHASE 3 : Modèle de base

## Étape 3.1 : Construction (`model_builder.py`)

 CNN From Scratch

Architecture typique :

```
Conv2D(32)

↓

MaxPooling

↓

Conv2D(64)

↓

MaxPooling

↓

Flatten

↓

Dense

↓

Dense(5)
```

Sortie :

```python
Dense(5, activation="softmax")
```

---

## Étape 3.2 : Compilation

```python
loss = SparseCategoricalCrossentropy()

optimizer = Adam(
    learning_rate=1e-3
)

metrics = ["accuracy"]
```

---

## Étape 3.3 : Entraînement (`train_phase3.py`)

TensorBoard :

```python
TensorBoard(
    log_dir="./logs"
)
```

Entraînement :

- 15 à 20 epochs

Toujours fournir :

```python
validation_data=
```

Sauvegarde :

```python
model.save(
    "models/baseline_model.keras"
)
```

---

## Étape 3.4 : Évaluation (`evaluate.py`)

Évaluation :

```python
model.evaluate(
    X_test,
    y_test
)
```

Visualisations :

- Accuracy
- Loss
- Matrice de confusion
- Courbes TensorBoard

Analyser :

- Underfitting
- Overfitting
- Objets confondus

---

# PHASE 4 : Optimisation

## Étape 4.1 : Réduction de l'Overfitting

### Data Augmentation

```python
RandomFlip("horizontal")

RandomRotation(0.1)

RandomZoom(0.1)
```

---

### Dropout

```python
Dropout(0.3)
```

ou

```python
Dropout(0.5)
```

---

### Early Stopping

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
```

---

## Étape 4.2 : Réduction de l'Underfitting

### Fine-Tuning

Dégeler les dernières couches :

```python
base_model.trainable = True
```

---

### Learning Rate faible

```python
Adam(
    learning_rate=1e-5
)
```

---

### Plus de capacité

Exemple :

```
Dense(256)

↓

Dense(128)

↓

Dense(5)
```

---

## Étape 4.3 : Nouvel entraînement

- 50 à 100 epochs
- EarlyStopping activé

Comparer dans TensorBoard :

- Phase 3
- Phase 4

Sauvegarder :

```python
model.save(
    "models/best_model.keras"
)
```

---

# PHASE 5 : Déploiement temps réel

## Étape 5.1 : Inférence webcam (`inference_realtime.py`)

Chargement :

```python
load_model(
    "models/best_model.keras"
)
```

Traitement de chaque image :

1. Capture webcam
2. Resize
3. Normalisation
4. float32
5. Ajout dimension batch

```python
np.expand_dims(...)
```

Prédiction :

```python
pred = model.predict(...)
```

Classe :

```python
np.argmax(pred)
```

Affichage OpenCV :

- Label
- Probabilité
- FPS

---

## Étape 5.2 : Stabilisation

### Moyenne mobile

Utiliser les **5 dernières prédictions**.

Objectif :

Limiter le scintillement.

---

### Seuil de confiance

Si :

```
max(probabilité) < 0.70
```

Afficher :

```
Background
```

ou

```
Incertain
```

---

## Étape 5.3 : Tests

Tester avec :

- faible luminosité
- forte luminosité
- contre-jour
- angles variés
- différentes distances
- objets partiellement cachés

Prévoir également :

- une vidéo enregistrée
- comme solution de secours pendant la soutenance

---

# PHASE 6 : Documentation & Soutenance

## Étape 6.1 : Rapport

### 1. Contexte

- Présentation du dataset
- Choix des objets
- Justification de Background

---

### 2. Architecture

Comparer :

- CNN
- Transfer Learning

Expliquer :

- convolution
- filtres
- MaxPooling

---

### 3. Journal d'expérimentation

Comparer :

- Phase 3
- Phase 4

Présenter :

- overfitting
- underfitting
- impact du Dropout
- impact du Fine-Tuning
- impact de la Data Augmentation
- impact de l'EarlyStopping

---

### 4. Visualisations

Inclure :

- Accuracy
- Loss
- TensorBoard
- Matrices de confusion

---

## Étape 6.2 : Démonstration

Démonstration :

- reconnaissance des 4 objets
- reconnaissance du Background
- affichage des probabilités
- fonctionnement en temps réel
