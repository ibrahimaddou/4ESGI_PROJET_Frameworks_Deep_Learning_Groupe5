import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Masquer les warnings TensorFlow pour garder un terminal propre
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_inference():
    print("Chargement du modèle en cours...")
    try:
        # Chargement du modèle que vous avez entraîné
        model = tf.keras.models.load_model('models/best_model.keras')
        print("Modèle chargé avec succès !")
    except Exception as e:
        print("Erreur : Impossible de charger le modèle. Vérifiez qu'il se trouve bien dans 'models/best_model.keras'.")
        return

    CLASS_NAMES = ["background", "badge", "briquet", "telephone"]

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    print("\n=== Lancement de la reconnaissance en temps réel ===")
    print("Appuyez sur la touche 'q' de votre clavier pour quitter le programme.")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la caméra.")
            break

        # 1. Calcul des FPS (Frames Per Second)
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Copie de la frame pour pouvoir dessiner dessus sans altérer l'image envoyée au modèle
        display_frame = frame.copy()

        # 2. Prétraitement de l'image (exactement comme dans training.py)
        # Redimensionnement à la taille attendue par le modèle (224x224)
        img_resized = cv2.resize(frame, (224, 224))
        
        # Conversion en tableau NumPy, ajout d'une dimension pour le "batch", et normalisation (0 à 1)
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # 3. Prédiction
        # verbose=0 permet de ne pas spammer le terminal à chaque frame
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class_name = CLASS_NAMES[predicted_class_idx]

        # 4. Affichage dynamique des couleurs et du texte
        # Choix de la couleur (en BGR pour OpenCV) en fonction du résultat
        if predicted_class_name == "background":
            color = (150, 150, 150)  # Gris si c'est le fond
        elif confidence > 0.85:
            color = (0, 255, 0)      # Vert si le modèle est très sûr de lui
        else:
            color = (0, 165, 255)    # Orange s'il hésite un peu

        # Dessin des textes sur l'image
        text_prediction = f"{predicted_class_name.upper()} ({confidence * 100:.1f}%)"
        cv2.putText(display_frame, text_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        text_fps = f"FPS: {int(fps)}"
        cv2.putText(display_frame, text_fps, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Affichage de la fenêtre
        cv2.imshow("Detection Temps Reel", display_frame)

        # Quitter si on appuie sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Fermeture de l'application...")
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()