"""
Module d'analyse de voies biologiques SBML et prédiction de l'essentialité génique via FlowGAT.
"""

import torch
import networkx as nx
import libsbml
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, global_mean_pool

class FlowGAT(torch.nn.Module):
    def __init__(self, num_node_features=1, hidden_dim=64, num_relations=1):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.lin   = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, data.batch)
        return torch.sigmoid(self.lin(x)).squeeze()

def load_pathway_sbml(sbml_file):
    """
    Lit un fichier SBML et construit un graphe NetworkX dirigé
    avec réactions et espèces comme noeuds.
    """
    reader = libsbml.SBMLReader()
    doc    = reader.readSBML(sbml_file)
    model  = doc.getModel()
    G = nx.DiGraph()

    # Ajout des noeuds réactions et des arêtes consommés/produits
    for reaction in model.getListOfReactions():
        rid = reaction.getId()
        G.add_node(rid, type="reaction")
        for reactant in reaction.getListOfReactants():
            sp = reactant.getSpecies()
            G.add_edge(sp, rid, type="consumes")
        for product in reaction.getListOfProducts():
            sp = product.getSpecies()
            G.add_edge(rid, sp, type="produces")
    return G

def graph_to_pyg_data(G):
    """
    Transforme un graphe NetworkX en Data PyG (noeuds uniformes, arêtes typées).
    """
    mapping = {n: i for i, n in enumerate(G.nodes())}
    edge_index, edge_type = [[], []], []
    for u, v, data in G.edges(data=True):
        edge_index[0].append(mapping[u])
        edge_index[1].append(mapping[v])
        edge_type.append(0)  # type unique pour simplicité

    x = torch.ones((len(G), 1), dtype=torch.float)
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_type=torch.tensor(edge_type, dtype=torch.long),
        batch=torch.zeros(len(G), dtype=torch.long)
    )

def load_model(path="models/flowgat_model.pt", device="cpu"):
    """
    Charge le modèle FlowGAT pré-entraîné.
    """
    model = FlowGAT(num_node_features=1)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_essentiality(sbml_file, model, device="cpu"):
    """
    Calcule un score d'essentialité pour chaque noeud du graphe SBML.
    Retourne un dictionnaire {noeud: score}.
    """
    G    = load_pathway_sbml(sbml_file)
    data = graph_to_pyg_data(G).to(device)
    with torch.no_grad():
        scores = model(data).cpu().numpy()
    return dict(zip(list(G.nodes()), scores))
