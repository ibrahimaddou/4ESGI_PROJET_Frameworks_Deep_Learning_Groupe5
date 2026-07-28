import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16

NUM_CLASSES = 4
IMAGE_SIZE = (224, 224, 3)

ARCHITECTURES = {
    "mobilenetv2": MobileNetV2,
    "resnet50":    ResNet50,
    "vgg16":       VGG16,
}

def build_model(
    architecture: str = "mobilenetv2",
    num_classes: int = NUM_CLASSES,
    fine_tune: bool = False,
    fine_tune_at: int = 100,
    learning_rate: float = 1e-4,
):
    """
    Construit un modèle de Transfer Learning.

    Params:
        architecture  : 'mobilenetv2', 'resnet50' ou 'vgg16'
        num_classes   : nombre de classes à prédire
        fine_tune     : si True, dégèle les couches à partir de fine_tune_at
        fine_tune_at  : indice de la couche à partir duquel on dégèle (fine-tuning)
        learning_rate : taux d'apprentissage pour l'optimiseur Adam

    Returns:
        model compilé prêt à l'entraînement
    """

    arch_key = architecture.lower()
    if arch_key not in ARCHITECTURES:
        raise ValueError(
            f"Architecture '{architecture}' inconnue. "
            f"Choisissez parmi : {list(ARCHITECTURES.keys())}"
        )

    base_fn = ARCHITECTURES[arch_key]

    # ── 1. CHARGEMENT DU MODÈLE DE BASE (poids ImageNet, sans la tête) ──────
    base_model = base_fn(
        input_shape=IMAGE_SIZE,
        include_top=False,       # On retire la couche Dense finale d'ImageNet
        weights="imagenet",
    )

    # ── 2. GEL DES COUCHES (Transfer Learning) ───────────────────────────────
    # On gèle toutes les couches : le modèle va juste affiner notre "tête"
    base_model.trainable = False

    # ── 3. FINE-TUNING (optionnel) ────────────────────────────────────────────
    # Si fine_tune=True, on dégèle les couches à partir de fine_tune_at
    # Utile une fois que la tête est bien entraînée (phase 2 d'entraînement)
    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(
            f"[Fine-Tuning] {arch_key} — "
            f"{len(base_model.layers) - fine_tune_at} couches dégelées "
            f"(sur {len(base_model.layers)} total)"
        )
    else:
        print(
            f"[Transfer Learning] {arch_key} — "
            f"toutes les couches gelées ({len(base_model.layers)} couches)"
        )

    # ── 4. CONSTRUCTION DE LA TÊTE (classification) ───────────────────────────
    inputs = tf.keras.Input(shape=IMAGE_SIZE)
    x = base_model(inputs, training=False)   # training=False => BN en inférence
    x = layers.GlobalAveragePooling2D()(x)   # Réduit (H, W, C) → (C,)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)              # Régularisation anti-overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"{arch_key}_transfer")

    # ── 5. COMPILATION ────────────────────────────────────────────────────────
    # Learning rate plus faible en fine-tuning pour ne pas détruire les poids
    lr = learning_rate / 10 if fine_tune else learning_rate

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nModèle '{model.name}' prêt.")
    print(f"  Paramètres entraînables : {model.trainable_variables.__len__():,} tenseurs")
    model.summary(line_length=80, show_trainable=True)

    return model


# ── TEST RAPIDE ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST model_builder.py")
    print("=" * 60)

    for arch in ["mobilenetv2", "resnet50", "vgg16"]:
        print(f"\n>>> Transfer Learning : {arch}")
        m = build_model(architecture=arch, fine_tune=False)
        print(f"    Output shape : {m.output_shape}")

    print("\n>>> Fine-Tuning : mobilenetv2 (couches > 100 dégelées)")
    m_ft = build_model(architecture="mobilenetv2", fine_tune=True, fine_tune_at=100)
    print(f"    Output shape : {m_ft.output_shape}")

    print("\n✅ model_builder.py OK !")