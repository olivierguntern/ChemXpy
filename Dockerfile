# Image de base Miniconda
FROM continuumio/miniconda3

# Copie du fichier environment
COPY env.yml /tmp/env.yml
RUN conda env create -f /tmp/env.yml

# Activation de l'environnement par défaut
SHELL ["conda", "run", "-n", "chem_graph_project", "/bin/bash", "-c"]

# Définition du répertoire de travail
WORKDIR /app

# Copie du code applicatif
COPY . /app

# Commande par défaut
CMD ["python", "main.py"]
