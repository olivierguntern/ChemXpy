"""
Script principal : orchestrateur du proof‑of‑concept.
Il exécute la rétro‑synthèse, la prédiction de rendement et l'analyse de voie.
"""

import os
import argparse
import pandas as pd
import plotly.express as px
from src.rxn_utils       import retrosynthesis
from src.yield_predict   import load_model as load_yield_model, predict_yield
from src.pathway_utils   import load_model as load_flowgat_model, predict_essentiality

def main():
    parser = argparse.ArgumentParser(description="Demo Graph AI chimie & biologie")
    parser.add_argument("--smiles", required=True, help="SMILES de la molécule cible")
    parser.add_argument("--sbml",   required=True, help="Fichier SBML de la voie métabolique")
    parser.add_argument("--output", default="results", help="Dossier de sortie")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("→ Lancement de la rétro‑synthèse")
    steps = retrosynthesis(args.smiles)
    reactions = [
        (",".join(step["reactants"]), ",".join(step["products"]))
        for step in steps
    ]

    print("→ Chargement du modèle de rendement")
    yield_model = load_yield_model()
    print("→ Prédiction des rendements")
    yields = predict_yield(reactions, yield_model)
    df_yield = pd.DataFrame({
        "étape":    [f"Étape {i+1}" for i in range(len(yields))],
        "rendement": yields
    })
    df_yield.to_csv(f"{args.output}/yields.csv", index=False)

    fig1 = px.bar(df_yield, x="étape", y="rendement",
                  title="Rendement estimé par étape")
    fig1.write_html(f"{args.output}/yield_plot.html")
    print("  • Rendements sauvegardés dans yields.csv et yield_plot.html")

    print("→ Analyse de l'essentialité génique")
    flowgat_model = load_flowgat_model()
    scores = predict_essentiality(args.sbml, flowgat_model)
    df_scores = pd.DataFrame(list(scores.items()), columns=["noeud", "score"])
    df_scores.to_csv(f"{args.output}/essentiality.csv", index=False)

    fig2 = px.scatter(df_scores, x="noeud", y="score",
                      title="Scores d'essentialité génique")
    fig2.write_html(f"{args.output}/essentiality_plot.html")
    print("  • Scores sauvegardés dans essentiality.csv et essentiality_plot.html")

    print("✓ Traitement terminé. Résultats disponibles dans le dossier", args.output)

if __name__ == "__main__":
    main()
