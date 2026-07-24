import os
import shutil
from sklearn.model_selection import train_test_split

def separer_donnees(dossier_source="data/raw", dossier_dest="data/processed", ratio_test=0.2):
    """
    Sépare les données brutes en jeux d'entraînement (80%) et de test (20%).
    """
    print(f"Début de la séparation des données (Test ratio: {ratio_test*100}%)...")
    
    dossier_train = os.path.join(dossier_dest, "train")
    dossier_test = os.path.join(dossier_dest, "test")
    
    os.makedirs(dossier_train, exist_ok=True)
    os.makedirs(dossier_test, exist_ok=True)
    
    # Parcourir chaque sous-dossier (classe d'objet) dans raw/
    classes = [d for d in os.listdir(dossier_source) if os.path.isdir(os.path.join(dossier_source, d))]
    
    for classe in classes:
        chemin_classe_source = os.path.join(dossier_source, classe)
        images = [f for f in os.listdir(chemin_classe_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            print(f"Attention: Aucune image trouvée dans {classe}.")
            continue
            
        # Split 80/20 classique
        train_imgs, test_imgs = train_test_split(images, test_size=ratio_test, random_state=42)
        
        # Création des sous-dossiers de destination
        os.makedirs(os.path.join(dossier_train, classe), exist_ok=True)
        os.makedirs(os.path.join(dossier_test, classe), exist_ok=True)
        
        # Copie des fichiers d'entraînement
        for img in train_imgs:
            shutil.copy2(os.path.join(chemin_classe_source, img), 
                         os.path.join(dossier_train, classe, img))
                         
        # Copie des fichiers de test
        for img in test_imgs:
            shutil.copy2(os.path.join(chemin_classe_source, img), 
                         os.path.join(dossier_test, classe, img))
                         
        print(f"Classe '{classe}' : {len(train_imgs)} images (Train), {len(test_imgs)} images (Test).")

if __name__ == "__main__":
    separer_donnees()