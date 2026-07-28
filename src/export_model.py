"""
export_model.py — Zakaria
Export du modèle final + tests de robustesse pour le rapport.

Tâches couvertes (répartition Phase 5) :
  - Exporter le modèle final en .keras et .h5
  - Tester la robustesse (luminosité, flou, bruit)
  - Lister les cas d'échec pour le rapport
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import json
import time

CLASS_NAMES  = ["background", "badge", "briquet", "telephone"]
MODEL_PATH   = "models/best_model.keras"
EXPORT_DIR   = "models/export"
TEST_DIR     = "dataset/processed/test"
REPORT_PATH  = "rapport/robustesse.json"


def load_model():
    print("Chargement du modèle...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Modèle chargé depuis '{MODEL_PATH}'")
        return model
    except Exception as e:
        print(f"  ❌ Erreur : {e}")
        print("  → Avez-vous lancé training.py ?")
        return None


def export_formats(model):
    """Export .keras (natif) et .h5 (legacy/compatibilité)."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    keras_path = os.path.join(EXPORT_DIR, "model_final.keras")
    model.save(keras_path)
    print(f"  ✅ Sauvegardé : {keras_path}")

    h5_path = os.path.join(EXPORT_DIR, "model_final.h5")
    model.save(h5_path)
    print(f"  ✅ Sauvegardé : {h5_path}")

    return keras_path, h5_path


def preprocess(img_array):
    """Prétraitement standard (identique à inference_realtime.py)."""
    img = cv2.resize(img_array, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx   = np.argmax(preds[0])
    return CLASS_NAMES[idx], float(preds[0][idx])


# ── PERTURBATIONS ──────────────────────────────────────────────────────────────
def perturb_brightness(img, factor):
    """Simule un éclairage fort ou faible."""
    return np.clip(img.astype("float32") * factor, 0, 255).astype("uint8")

def perturb_blur(img, ksize):
    """Simule une image floue (netteté insuffisante)."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def perturb_noise(img, sigma):
    """Simule du bruit de capteur."""
    noise = np.random.normal(0, sigma, img.shape).astype("float32")
    return np.clip(img.astype("float32") + noise, 0, 255).astype("uint8")


def robustness_tests(model):
    """
    Teste le modèle sur les images du jeu de test avec des perturbations.
    Retourne un dict de résultats pour le rapport.
    """
    datagen = ImageDataGenerator(rescale=1.0)   # On ne normalise pas ici, on le fait à la main
    gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
    )

    scenarios = {
        "normal":           lambda img: img,
        "bright_x1.5":     lambda img: perturb_brightness(img, 1.5),
        "dark_x0.4":        lambda img: perturb_brightness(img, 0.4),
        "blur_k15":         lambda img: perturb_blur(img, 15),
        "noise_sigma30":    lambda img: perturb_noise(img, 30),
    }

    results = {name: {"correct": 0, "total": 0, "failures": []} for name in scenarios}

    true_classes = list(gen.class_indices.keys())

    print(f"\n  Test sur {gen.samples} images × {len(scenarios)} scénarios...")

    for i in range(gen.samples):
        batch_img, batch_label = next(gen)
        raw_img = (batch_img[0] * 255).astype("uint8")   # Revenir en uint8 pour cv2
        true_idx   = np.argmax(batch_label[0])
        true_class = true_classes[true_idx]
        img_path   = gen.filepaths[i]

        for name, fn in scenarios.items():
            perturbed   = fn(raw_img)
            processed   = preprocess(perturbed)
            pred_class, confidence = predict(model, processed)

            results[name]["total"] += 1
            if pred_class == true_class:
                results[name]["correct"] += 1
            else:
                results[name]["failures"].append({
                    "image":       os.path.basename(img_path),
                    "true":        true_class,
                    "predicted":   pred_class,
                    "confidence":  round(confidence, 3),
                })

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n  Résultats de robustesse :")
    print(f"  {'Scénario':<22} {'Accuracy':>10}  {'Échecs':>8}")
    print("  " + "-" * 45)
    for name, r in results.items():
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {name:<22} {acc:>9.1f}%  {len(r['failures']):>7}")

    return results


def benchmark_inference_speed(model, n_runs=50):
    """Mesure la vitesse d'inférence (FPS moyen)."""
    dummy = np.random.rand(1, 224, 224, 3).astype("float32")
    model.predict(dummy, verbose=0)   # Warm-up

    start = time.time()
    for _ in range(n_runs):
        model.predict(dummy, verbose=0)
    elapsed = time.time() - start

    fps = n_runs / elapsed
    ms  = (elapsed / n_runs) * 1000
    print(f"\n  Vitesse d'inférence : {fps:.1f} FPS  ({ms:.1f} ms / image, n={n_runs})")
    return {"fps": round(fps, 2), "ms_per_image": round(ms, 2)}


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EXPORT & ROBUSTESSE — Zakaria")
    print("=" * 60)

    model = load_model()
    if model is None:
        exit(1)

    # Export
    print("\n► Export du modèle")
    keras_path, h5_path = export_formats(model)

    # Benchmark vitesse
    print("\n► Benchmark vitesse d'inférence")
    speed = benchmark_inference_speed(model)

    # Tests de robustesse
    print("\n► Tests de robustesse")
    robustness = robustness_tests(model)

    # Sauvegarde rapport JSON (pour le rapport de soutenance)
    os.makedirs("rapport", exist_ok=True)
    report = {
        "model_paths": {"keras": keras_path, "h5": h5_path},
        "inference_speed": speed,
        "robustness": {
            k: {
                "accuracy_pct": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
                "total": v["total"],
                "failures_count": len(v["failures"]),
                "failures_sample": v["failures"][:5],   # 5 exemples max
            }
            for k, v in robustness.items()
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Rapport de robustesse sauvegardé : '{REPORT_PATH}'")
    print("   Utilisez ce fichier pour rédiger la section 'tests de robustesse' du rapport.")