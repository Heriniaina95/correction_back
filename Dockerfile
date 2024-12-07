# Étape 1: Utiliser une image de base Python
FROM python:3.11-slim

# Étape 2: Mettre à jour les paquets système et installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Étape 3: Installer Rust (nécessaire pour certaines dépendances Python)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$PATH:$HOME/.cargo/bin"

# Étape 4: Mettre à jour pip
RUN pip install --upgrade pip

# Étape 5: Installer les dépendances Python nécessaires
RUN pip install --no-cache-dir \
    Flask==2.2.3 \
    Werkzeug==2.2.3 \
    torch \
    sentencepiece \
    transformers==4.33.0 \
    sacremoses \
    numpy \
    requests \
    flask-cors

# Étape 6: Définir le répertoire de travail
RUN mkdir /app
WORKDIR /app

# Étape 7: Copier les dossiers des modèles dans le conteneur
COPY model_fr_en/ /app/model_fr_en
COPY model_en_fr/ /app/model_en_fr

# Étape 8: Copier tout le code de l'application dans le conteneur
COPY . /app

# Étape 9: Exposer le port 5000
EXPOSE 5000

# Étape 10: Lancer l'application Flask (serveur de développement de Flask)
CMD ["python", "app.py"]
