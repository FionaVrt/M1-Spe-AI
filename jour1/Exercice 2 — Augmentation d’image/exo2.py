# === AUGMENTATION D'IMAGES ===


import cv2  # Bibliothèque de vision par ordinateur
import numpy as np  # Opérations numériques
import matplotlib.pyplot as plt  # Visualisation

# === CHARGEMENT ET PRÉTRAITEMENT DE L'IMAGE ===
# Charger l'image depuis le fichier
img = cv2.imread("image.jpg")

# Convertir de BGR (format OpenCV) à RGB (format standard)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normaliser les valeurs de pixels (0-255) en (0-1) pour les calculs
img = img / 255.0

# === FONCTION D'AUGMENTATION D'IMAGE ===
def augment(img):
    """
    Applique plusieurs transformations aléatoires à une image
    
    Args:
        img: Image numpy array en format RGB normalisée (0-1)
        
    Returns:
        tuple: (image augmentée, liste des modifications appliquées)
    """
    mods = []  # Liste pour enregistrer les modifications appliquées

    # === FLIP HORIZONTAL ===
    # Retourner l'image horizontalement avec 50% de probabilité
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
        mods.append("Flip horizontal")

    # === ROTATION LÉGÈRE ===
    # Appliquer une rotation aléatoire entre -10° et +10°
    angle = np.random.uniform(-10, 10)  # Angle aléatoire
    h, w, _ = img.shape  # Obtenir les dimensions
    
    # Créer la matrice de rotation
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    
    # Appliquer la rotation à l'image
    img = cv2.warpAffine(img, M, (w, h))
    mods.append(f"Rotation {angle:.1f}°")

    # === BRUIT GAUSSIEN ===
    # Ajouter du bruit aléatoire pour simuler des conditions réelles
    noise = np.random.normal(0, 0.02, img.shape)  # Bruit gaussien
    
    # Ajouter le bruit et limiter les valeurs entre 0 et 1
    img = np.clip(img + noise, 0, 1)
    mods.append("Bruit gaussien")

    return img, mods

# === AUGMENTATION ===
# Appliquer les transformations aléatoires
aug_img, modifications = augment(img)

# === VISUALISATION ===
# Créer une figure avec deux sous-graphiques
plt.figure(figsize=(8, 4))

# Sous-graphique 1: Image originale
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

# Sous-graphique 2: Image augmentée
plt.subplot(1, 2, 2)
plt.imshow(aug_img)
plt.title("Augmentée\n" + ", ".join(modifications))
plt.axis("off")

# Afficher la figure
plt.show()
