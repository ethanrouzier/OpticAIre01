# Étape 1 : Image de base Python
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier les requirements d'abord (optimisation du cache)
COPY requirements.txt .

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier le reste de l'application
COPY . .

# Étape 6 : Exposer le port
EXPOSE 5000

# Étape 7 : Commande de démarrage avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]