FROM python:3.9

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt ./
COPY app.py ./
COPY app_monitoring1.py ./
COPY app_monitoring2.py ./
COPY catboost_model-1.pkl ./

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port Flask
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
