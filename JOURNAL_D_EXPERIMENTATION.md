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