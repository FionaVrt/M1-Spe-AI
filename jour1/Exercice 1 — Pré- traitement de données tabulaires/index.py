# === Chargement des données ===
import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("data.csv")
print(df.head())

# === Nettoyage des données ===
# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Supprimer les doublons
df = df.drop_duplicates()

# === Séparation des features et de la cible ===
# X : features, y : variable cible (remplace "target" par le nom réel)
X = df.drop("target", axis=1)
y = df["target"]

# === Imputation des valeurs manquantes ===
from sklearn.impute import SimpleImputer

# Imputer les valeurs manquantes avec la moyenne pour les variables numériques
imputer = SimpleImputer(strategy="mean")
X_num = imputer.fit_transform(X.select_dtypes(include="number"))

# === Normalisation des données numériques ===
from sklearn.preprocessing import StandardScaler

# Normaliser les features numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# === Encodage des variables catégorielles ===
# Convertir les variables catégorielles en variables numériques (one-hot encoding)
X_cat = pd.get_dummies(X.select_dtypes(include="object"))

# === Séparation en train, validation et test ===
from sklearn.model_selection import train_test_split

# Première séparation : 70% train, 30% temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Deuxième séparation : diviser temp en 50% validation et 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
