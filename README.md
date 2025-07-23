# Projet Graph AI en Chimie & Biologie

Ce projet propose un démonstrateur léger de bout en bout :

- Rétro‑synthèse via l'API IBM RXN for Chemistry  
- Prédiction de rendement avec un GNN pré‑entraîné  
- Analyse de l'essentialité génique dans des voies SBML avec FlowGAT  

## Prérequis

1. Clé API IBM RXN : définir `IBMRXN_TOKEN`  
2. Modèles pré‑entraînés dans `models/` :  
   - `yield_model.pt`  
   - `flowgat_model.pt`

## Installation

```bash
git clone <url_du_dépôt>
cd project
conda env create -f env.yml
conda activate chem_graph_project
```

## Utilisation

```bash
python main.py --smiles "CCO" --sbml data/pathway.sbml --output results
```

Les résultats (CSV et graphiques HTML) seront générés dans le dossier `results`.
