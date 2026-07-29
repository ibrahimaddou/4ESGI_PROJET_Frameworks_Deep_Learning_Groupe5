# Journal d'Expérimentation
## Projet : Reconnaissance d'Objets du Quotidien en Temps Réel

---

# PHASE 1 : PRÉPARATION

## Expérience 1.1 : Sélection des Objets

### Description
Choix et justification des objets pour la classification.

## Experiment 1.2: Training the Base Model

### Analyse des résultats du modèle

*   **Observation :** 
    En observant les courbes d'apprentissage sur TensorBoard (voir Capture d'écran tensorboard_1.2 dans le dossier logs/tensorboard), nous remarquons que la *Train Accuracy* et la *Val Accuracy* convergent très rapidement vers un score presque parfait (environ 99%). De plus, la matrice de confusion (voir Capture d'écran figure_1.2 dans le dossier logs/tensorboard) confirme cette excellente performance : le modèle a classé correctement 119 images de test sur 120 (soit un score de 30/30 pour le badge, le briquet et le téléphone).
    
*   **Analyse (Biais et Variance) :** 
    Le modèle ne souffre d'**aucun problème de Biais (Underfitting)** car sa précision globale est très élevée. Il ne souffre pas non plus de **Variance (Overfitting)**, car la courbe de validation suit parfaitement la courbe d'entraînement sans jamais décrocher. L'apprentissage est très stable.

*   **Analyse des erreurs :**
    La matrice de confusion nous montre une seule erreur marginale : un faux positif où le modèle a prédit "téléphone" alors qu'il s'agissait du "background" (1 erreur sur 30). Cela peut s'expliquer par un reflet, une ombre ou un élément de texture dans le fond de l'image qui a été interprété comme la surface lisse du téléphone.
## Expérience 1.3 : Ajout de Data Augmentation et Dropout

### Description
Pour corriger l'Overfitting observé lors de l'expérience 1.2, nous avons ajouté des transformations aléatoires sur les images (**Data Augmentation**) telles que des rotations et des zooms, ainsi qu'une couche **Dropout** avec un taux de 0.5.

### Analyse des résultats du modèle

* **Observation :**
L'Overfitting diminue fortement. La précision en validation remonte à environ **82 %**, mais elle n'arrive plus à progresser au-delà malgré les epochs supplémentaires.

* **Analyse (Biais et Variance) :**
Le modèle est désormais beaucoup plus stable. La **Variance** est fortement réduite grâce aux techniques de régularisation. En revanche, les performances plafonnent, ce qui indique un léger **Biais (Underfitting)** : un petit CNN entraîné depuis zéro ne possède pas une capacité suffisante pour extraire les caractéristiques visuelles complexes des objets, notamment les surfaces brillantes ou présentant des reflets.

* **Conclusion :**
La régularisation améliore la généralisation mais révèle les limites de l'architecture utilisée. Nous décidons donc d'utiliser le **Transfer Learning** afin de bénéficier d'un réseau profond pré-entraîné sur plusieurs millions d'images.

---

## Expérience 1.4 : Premier test de Transfer Learning (MobileNetV2)

### Analyse des résultats du modèle

* **Observation :**
En observant les courbes d'apprentissage sur TensorBoard (voir capture **tensorboard_1.2** dans le dossier `logs/tensorboard`), nous remarquons que la **Train Accuracy** et la **Validation Accuracy** convergent très rapidement vers un score proche de **99 %**. De plus, la matrice de confusion (voir capture **figure_1.2** dans le dossier `logs/tensorboard`) confirme cette excellente performance : le modèle classe correctement **119 images sur 120**, soit un score de **30/30** pour les classes **badge**, **briquet** et **téléphone**.

* **Analyse (Biais et Variance) :**
Le modèle ne présente **aucun problème de Biais (Underfitting)** puisque sa précision globale est très élevée. Il ne souffre pas non plus de **Variance (Overfitting)** : les courbes d'entraînement et de validation restent très proches durant tout l'apprentissage, ce qui traduit une excellente capacité de généralisation.

* **Analyse des erreurs :**
La matrice de confusion révèle une seule erreur marginale : une image de **background** est prédite comme **téléphone**. Cette confusion peut s'expliquer par un reflet, une ombre ou une texture présente dans le fond de l'image qui a été interprétée comme la surface lisse et brillante d'un téléphone.

---

## Experiment 2 : Modèle Final
#### 1. Analyse des performances et du Biais / Variance
* **Observation (TensorBoard) :** L'entraînement du modèle `mobilenetv2_transfer` (avec couches de base gelées et ajout d'une couche Dense de 256 neurones + Dropout 0.3) montre une convergence exceptionnelle. La perte de validation (*val_loss*) descend continuellement jusqu'à **0.0062** à l'Epoch 20 (voir capture `epoch_loss`), tandis que la précision (*val_accuracy*) atteint **100 %** (voir capture `epoch_accuracy`).
* **Diagnostic :** 
  * **Absence de Biais (Underfitting) :** Le modèle a une capacité d'abstraction largement suffisante pour différencier les 4 classes dès la 3ᵉ epoch.
  * **Absence de Variance (Overfitting) :** Contrairement à un réseau classique qui mémorise, l'association du **Transfer Learning** (poids ImageNet) et de la couche de **Dropout (0.3)** empêche le modèle d'apprendre par cœur. Les courbes *Train* et *Validation* restent parfaitement jointives.

#### 2. Analyse de la Matrice de Confusion
Sur le jeu de test de 120 images (30 images par classe), le modèle obtient un score de **100 % de précision et de rappel** (F1-score de 1.00 sur toutes les classes). L'ancienne confusion marginale entre le *background* et le *téléphone* a été résolue grâce à la régularisation de la nouvelle architecture de classification.

#### 3. Tests de Robustesse et Limites du Modèle
Pour évaluer les limites réelles du système en conditions dégradées, nous avons soumis le jeu de test à 4 perturbations artificielles (`src/export_model.py`) :
* **Sensibilité à l'éclairage (65.0 % en surexposition / 52.5 % en sous-exposition) :** Lorsque la luminosité augmente fortement (`bright_x1.5`), le modèle échoue principalement sur la classe **badge** qu'il confond avec le **téléphone** (ex: `badge_009.jpg` prédit *téléphone* avec 67.4 % de confiance). Cela s'explique par l'apparition de reflets blancs sur le plastique du badge, rappelant la surface brillante de l'écran du téléphone.
* **Sensibilité à la netteté (27.5 % avec flou gaussien `k=15`) :** Lorsque les arêtes sont lissées, le modèle perd ses repères spatiaux et classe la majorité des objets comme du **background** (ex: `badge_018.jpg` prédit *background* à 68 %).
* **Vitesse d'inférence :** Le modèle s'exécute à une moyenne de **10.6 FPS** (~94.3 ms/image) sur processeur (CPU), ce qui est largement suffisant pour une détection fluide en temps réel via webcam.
