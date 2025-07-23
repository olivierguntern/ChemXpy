"""
Module d'interaction avec l'API IBM RXN for Chemistry.
Permet d'obtenir automatiquement une route de synthèse pour une molécule donnée.
"""

import os
import time
import requests

API_BASE   = "https://rxn.res.ibm.com/rxn/"
API_RETRO  = API_BASE + "api/reactions"

def retrosynthesis(smiles, max_retries=60, retry_interval=5):
    """
    Envoie un SMILES au service IBM RXN et récupère la séquence de réactions proposée.
    Renvoie une liste de règles de transformation réactifs->produits.
    """
    token = os.getenv("IBMRXN_TOKEN")
    if not token:
        raise ValueError("La variable d'environnement IBMRXN_TOKEN doit être définie.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {"smiles": smiles}

    # Envoi de la requête initiale
    response = requests.post(API_RETRO, json=payload, headers=headers)
    response.raise_for_status()
    job_id = response.json().get("id")

    # Boucle d'attente jusqu'à obtention du résultat
    for _ in range(max_retries):
        status = requests.get(f"{API_RETRO}/{job_id}", headers=headers)
        status.raise_for_status()
        result = status.json()
        if result.get("status") == "completed":
            return result.get("template_reaction_rules", [])
        time.sleep(retry_interval)

    raise TimeoutError("Délai dépassé lors de la rétro-synthèse.")
