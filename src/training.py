import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les warnings et infos de TensorFlow 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Désactive le message oneDNN spécifié dans votre terminal 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import datetime

# from model_builder import build_model 
##############################################################################
from model_builder import build_model
##############################################################################

def train_model():
    # 1. PRÉPARATION DES DONNÉES ET NORMALISATION
    print("Configuration des générateurs de données...")
    
    # Le rescale 1./255 ramène les valeurs des pixels entre 0 et 1
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0, 
        # Data Augmentation to add here(rotation, zoom, etc.)
    )
    
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        'dataset/processed/train',
        target_size=(224, 224), # Taille standard requise par VGG16/MobileNet
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        'dataset/processed/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # 2. CHARGEMENT DU MODÈLE 
    # model = build_model(num_classes=4) 
    # Pour tester le script avant que Zakaria ait fini, on utilise un modèle vide temporaire :
    ##############################################################################
    model = build_model(
        architecture="mobilenetv2",
        num_classes=train_generator.num_classes,
        fine_tune=False,
        learning_rate=1e-4,
    )
    ##############################################################################
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(224, 224, 3)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(4, activation='softmax')
    # ])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. CONFIGURATION DES CALLBACKS CRUCIAUX
    # TensorBoard : Enregistre les métriques pour les graphiques du rapport
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # EarlyStopping : Stoppe l'entraînement si le modèle commence à faire de l'overfitting (Variance)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, # Attendre 5 epochs sans amélioration avant de couper
        restore_best_weights=True # Garder la meilleure version du modèle
    )

    # ModelCheckpoint : Sauvegarde automatiquement le meilleur modèle
    os.makedirs('models', exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )

    # 4. LANCEMENT DE L'ENTRAÎNEMENT
    print("Début de l'entraînement...")
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=test_generator,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        verbose=1
    )
    print("Entraînement terminé. Le meilleur modèle est sauvegardé dans 'models/best_model.keras // .h5'.")

if __name__ == "__main__":
    train_model()