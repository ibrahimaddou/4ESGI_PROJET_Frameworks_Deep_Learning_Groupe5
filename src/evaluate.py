import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les warnings et infos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Désactive le message oneDNN spécifié dans votre terminal
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    print("Chargement du modèle et des données de test...")
    
    # 1. CHARGEMENT DU MODÈLE SAUVEGARDÉ
    try:
        model = tf.keras.models.load_model('models/best_model.keras')
    except Exception as e:
        print("Erreur : Impossible de charger le modèle. Avez-vous lancé training.py ?")
        return

    # 2. PRÉPARATION DES DONNÉES DE TEST
    # Important : shuffle=False pour que les prédictions correspondent aux vraies étiquettes
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        'dataset/processed/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False 
    )

    # Récupération des vraies classes et de leurs noms
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    # 3. PRÉDICTIONS
    print("Génération des prédictions en cours...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)

    # 4. RAPPORT DE CLASSIFICATION
    print("\n=== RAPPORT DE CLASSIFICATION ===")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 5. MATRICE DE CONFUSION
    print("\n=== GÉNÉRATION DE LA MATRICE DE CONFUSION ===")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    # Sauvegarde de l'image 
    os.makedirs('rapport/images', exist_ok=True)
    plt.savefig('rapport/images/matrice_confusion.png')
    print("Matrice de confusion sauvegardée dans 'rapport/images/matrice_confusion.png'.")
    plt.show()

if __name__ == "__main__":
    evaluate_model()