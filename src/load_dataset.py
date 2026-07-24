import os
from PIL import Image
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = "dataset/processed"
IMAGE_SIZE = (224, 224)

def check_dataset_structure():
    """Vérifie que la structure des données est correcte"""
    print("\n  Vérification structure dataset...\n")
    
    for split in ['train', 'test']:
        split_dir = os.path.join(PROCESSED_DATA_DIR, split)
        
        if not os.path.exists(split_dir):
            print(f" {split}: MANQUANT")
            continue
        
        print(f" {split}:")
        total_images = 0
        
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                num_images = len(os.listdir(class_dir))
                total_images += num_images
                print(f"   - {class_name}: {num_images} images")
        
        print(f"   TOTAL: {total_images} images\n")

def get_dataset_stats():
    """Affiche les stats du dataset"""
    print("\n Statistiques Dataset\n")
    
    for split in ['train', 'test']:
        split_dir = os.path.join(PROCESSED_DATA_DIR, split)
        
        if not os.path.exists(split_dir):
            continue
        
        class_counts = {}
        total = 0
        
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                class_counts[class_name] = count
                total += count
        
        print(f" {split.upper()} ({total} images):")
        for class_name, count in sorted(class_counts.items()):
            if total > 0:
                percentage = (count / total) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        print()

def visualize_sample_images(num_samples=9):
    """Affiche quelques images du dataset pour vérifier"""
    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    
    if not os.path.exists(train_dir):
        print(" Dossier train introuvable!")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    
    count = 0
    for class_name in sorted(os.listdir(train_dir)):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if count >= num_samples:
                    break
                
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = Image.open(img_path)
                    axes[count].imshow(img)
                    axes[count].set_title(class_name)
                    axes[count].axis('off')
                    count += 1
                except:
                    pass
        
        if count >= num_samples:
            break
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=100)
    print(" Images sauvegardées dans 'dataset_samples.png'")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES KERAS")
    print("=" * 60)
    
    check_dataset_structure()
    get_dataset_stats()
    
    print("\n Génération aperçu images...")
    visualize_sample_images()
    
    print("\n" + "=" * 60)
    print(" VALIDATION DATASET COMPLÈTE!")
    print("=" * 60)