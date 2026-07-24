import cv2 
import os
import time
import sys

# LISTE DE TES OBJETS 
CLASSES_TO_CAPTURE = ["briquet", "badge", "telephone", "background"]
MAX_IMAGES = 150
INTERVAL_SEC = 0.2  # Temps entre chaque capture (200ms)

# Initialisation de la webcam (une seule fois pour tout le script)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'accéder à la webcam.")
    sys.exit()

print("Démarrage du programme de capture global...")

# Boucle pour passer d'un objet à l'autre
for CLASS_NAME in CLASSES_TO_CAPTURE:
    SAVE_DIR = os.path.join("dataset", CLASS_NAME)
    
    # Création du dossier spécifique à l'objet s'il n'existe pas
    os.makedirs(SAVE_DIR, exist_ok=True)

    count = 0
    capturing = False
    last_capture_time = time.time()

    print("\n" + "="*50)
    print(f"=== PRÉPAREZ L'OBJET : {CLASS_NAME.upper()} ===")
    print("="*50)
    print("Instructions :")
    print("  - Appuyez sur 'c' pour DÉMARRER / PAUSER la capture.")
    print("  - Appuyez sur 'n' pour PASSER à l'objet suivant (si fini avant).")
    print("  - Appuyez sur 'q' pour QUITTER complètement le programme.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la lecture du flux vidéo.")
            break

        current_time = time.time()

        # Capture automatique si activée
        if capturing and (current_time - last_capture_time >= INTERVAL_SEC):
            if count < MAX_IMAGES:
                img_name = f"{CLASS_NAME}_{count:03d}.jpg"
                img_path = os.path.join(SAVE_DIR, img_name)
                cv2.imwrite(img_path, frame)
                count += 1
                last_capture_time = current_time
                print(f"[{count}/{MAX_IMAGES}] Image enregistrée : {img_path}")
            else:
                capturing = False
                print(f"\n Capture terminée pour '{CLASS_NAME}' ! On passe au suivant.")
                break # Sort de la boucle pour passer à l'objet suivant

        # Affichage des infos sur le flux vidéo
        status_text = f"{CLASS_NAME.upper()} - Capturing: {count}/{MAX_IMAGES}" if capturing else f"{CLASS_NAME.upper()} - PAUSED ({count}/{MAX_IMAGES})"
        color = (0, 255, 0) if capturing else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Capture Dataset", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Gestion des touches clavier
        if key == ord('c'):
            capturing = not capturing
        elif key == ord('n'):
            print(f"Passage forcé à l'objet suivant...")
            break # Sort de la boucle pour passer à l'objet suivant
        elif key == ord('q'):
            print("Fermeture complète du programme...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit() # Quitte le script complètement

# Quand tous les objets de la liste sont passés
cap.release()
cv2.destroyAllWindows()
print("\n🎉 TOUTES LES CAPTURES SONT TERMINÉES ! Vous pouvez vérifier le dossier 'dataset/'.")