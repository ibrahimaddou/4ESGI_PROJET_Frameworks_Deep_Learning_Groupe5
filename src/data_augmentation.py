from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TRAIN_DIR = "dataset/processed/train"
VAL_DIR = "dataset/processed/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# GÉNÉRATEUR D'ENTRAÎNEMENT AVEC AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# GÉNÉRATEUR VALIDATION SANS AUGMENTATION
val_datagen = ImageDataGenerator(rescale=1./255)

def get_train_generator():
    """Retourne générateur train avec augmentation"""
    if not os.path.exists(TRAIN_DIR):
        print(f" Erreur : {TRAIN_DIR} n'existe pas!")
        return None
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    print(f" Train Generator prêt - {train_gen.samples} images")
    print(f"   Classes: {train_gen.class_indices}")
    return train_gen

def get_val_generator():
    """Retourne générateur val sans augmentation"""
    if not os.path.exists(VAL_DIR):
        print(f" Erreur : {VAL_DIR} n'existe pas!")
        return None
    
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f" Val Generator prêt - {val_gen.samples} images")
    return val_gen

if __name__ == "__main__":
    print("\n🔄 Test Data Augmentation\n")
    train_gen = get_train_generator()
    val_gen = get_val_generator()
    
    if train_gen and val_gen:
        print("\n Tous les générateurs fonctionnent!")
    else:
        print("\n Erreur lors du chargement")