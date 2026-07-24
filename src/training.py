import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. DÉFINITION DES GÉNÉRATEURS
print("Préparation des générateurs de données...")

# La division par 255.0 s'effectue via le paramètre 'rescale'
generateur_train = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalisation cruciale
    rotation_range=20,    # Data augmentation
    horizontal_flip=True
)

generateur_test = ImageDataGenerator(
    rescale=1.0 / 255.0   # Le jeu de test doit OBLIGATOIREMENT être normalisé de la même manière
)

# 2. CHARGEMENT DES IMAGES DEPUIS LES DOSSIERS
# (Assure-toi que les chemins correspondent à ce que ton script de split a généré)
dossier_train = "data/processed/train"
dossier_test = "data/processed/test"

train_data = generateur_train.flow_from_directory(
    dossier_train,
    target_size=(224, 224), # Taille standard pour MobileNet/VGG16
    batch_size=32,
    class_mode='categorical'
)

test_data = generateur_test.flow_from_directory(
    dossier_test,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)