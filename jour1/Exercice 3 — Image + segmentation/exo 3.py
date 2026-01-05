# === SEGMENTATION D'IMAGE ===
# Ce script crée une image avec des formes géométriques et génère un masque de segmentation

import cv2  # Bibliothèque de vision par ordinateur
import numpy as np  # Opérations numériques
import matplotlib.pyplot as plt  # Visualisation

# === CRÉATION DE L'IMAGE ===
# Créer une image blanche de 256x256 pixels en RGB
img = np.ones((256, 256, 3), dtype=np.uint8) * 255

# === DESSINER DES FORMES ===
# Dessiner un rectangle bleu (BGR: 255,0,0 = bleu)
# Paramètres: image, point haut-gauche, point bas-droit, couleur BGR, -1 = rempli
cv2.rectangle(img, (30, 30), (150, 120), (255, 0, 0), -1)

# Dessiner un cercle vert (BGR: 0,255,0 = vert)
# Paramètres: image, centre, rayon, couleur BGR, -1 = rempli
cv2.circle(img, (180, 180), 40, (0, 255, 0), -1)

# === CRÉATION DU MASQUE DE SEGMENTATION ===
# Créer un masque noir (tous les pixels à 0)
mask = np.zeros((256, 256), dtype=np.uint8)

# Marquer le rectangle sur le masque (valeur 1 = objet)
cv2.rectangle(mask, (30, 30), (150, 120), 1, -1)

# Marquer le cercle sur le masque (valeur 1 = objet)
cv2.circle(mask, (180, 180), 40, 1, -1)

# === VISUALISATION ===
# Créer une figure avec deux sous-graphiques
plt.figure(figsize=(8, 4))

# Sous-graphique 1: Image originale
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Image RGB")
plt.axis("off")

# Sous-graphique 2: Masque de segmentation
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Masque de segmentation")
plt.axis("off")

# Afficher la figure
plt.show()
