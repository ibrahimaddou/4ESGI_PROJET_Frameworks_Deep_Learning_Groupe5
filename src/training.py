import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Masque les warnings et infos de TensorFlow
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = (
    "0"  # Désactive le message oneDNN spécifié dans votre terminal
)

import tensorflow as tf
from model_builder import build_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model():
    # ── 1. PRÉPARATION DES DONNÉES ET DATA AUGMENTATION ─────────────────────
    print("Configuration des générateurs de données...")

    # Ajout de la Data Augmentation (pour justifier notre rapport)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        "dataset/processed/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    test_generator = test_datagen.flow_from_directory(
        "dataset/processed/test",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    # ── 2. CALLBACKS ────────────────────────────────────────────────────────
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    os.makedirs("models", exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath="models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    )

    # ── 3. PHASE 1 : TRANSFER LEARNING (Couches de base gelées) ─────────────
    print("\n>>> [PHASE 1] Transfer Learning : Entraînement de la tête...")
    model = build_model(
        architecture="mobilenetv2",
        num_classes=train_generator.num_classes,
        fine_tune=False,
        learning_rate=1e-4,
    )

    history_tl = model.fit(
        train_generator,
        epochs=15,  # 15 epochs suffisent pour stabiliser la tête
        validation_data=test_generator,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        verbose=1,
    )

    # ── 4. PHASE 2 : FINE-TUNING (Dégel des couches profondes) ──────────────
    print("\n>>> [PHASE 2] Fine-Tuning : Affinage des couches profondes...")

    # 1. On dégèle l'ensemble du modèle principal
    model.trainable = True

    # 2. On récupère dynamiquement le modèle de base (ex: MobileNetV2)
    #    Il s'agit de la couche qui contient elle-même des sous-couches
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    # 3. On gèle les 100 premières couches du modèle de base (Fine-tuning des couches profondes)
    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
        print(
            f" -> {len(base_model.layers) - 100} couches dégelées sur {len(base_model.layers)} dans le modèle de base."
        )
    else:
        # Fallback de sécurité si l'architecture n'est pas imbriquée
        for layer in model.layers[: -3]:  # On garde seulement la tête dégelée
            layer.trainable = False

    # IMPORTANT : Le learning rate doit être réduit (ex: divisé par 10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # On remet un checkpoint spécifique au fine-tuning
    ft_checkpoint = ModelCheckpoint(
        filepath="models/best_model_finetuned.keras",
        monitor="val_accuracy",
        save_best_only=True,
    )

    history_ft = model.fit(
        train_generator,
        epochs=10,  # 10 epochs supplémentaires pour affiner
        validation_data=test_generator,
        callbacks=[tensorboard_callback, early_stopping, ft_checkpoint],
        verbose=1,
    )

    print("\nEntraînement en deux étapes terminé avec succès !")
    print(" 1. Modèle Transfer Learning  -> models/best_model.keras")
    print(" 2. Modèle Fine-Tuné (FINAL) -> models/best_model_finetuned.keras")


if __name__ == "__main__":
    train_model()