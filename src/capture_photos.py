import cv2 
import os
import time

CLASS_NAME = "badge"
SAVE_DIR = os.path.join("dataset", CLASS_NAME)
MAX_IMAGES = 150
INTERVAL_SEC = 0.2  # Temps entre chaque capture (200ms)

# Création du dossier s'il n'existe pas
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'accéder à la webcam.")
    exit()

count = 0
capturing = False
last_capture_time = time.time()

print(f"=== Capture pour la classe : {CLASS_NAME} ===")
print("Instructions :")
print("  - Appuyez sur 'c' pour DÉMARRER / PAUSER la capture automatique.")
print("  - Appuyez sur 'q' pour QUITTER.")

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
            print("Capture terminée ! Target atteinte.")

    # Affichage des infos sur le flux vidéo
    status_text = f"Capturing: {count}/{MAX_IMAGES}" if capturing else f"PAUSED ({count}/{MAX_IMAGES})"
    color = (0, 255, 0) if capturing else (0, 0, 255)
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Capture Dataset - Badge", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        capturing = not capturing
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()