"""
Module de prédiction de rendement de réaction à l'aide d'un GNN pré-entraîné.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

class YieldGNN(torch.nn.Module):
    def __init__(self, num_node_features=1, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin   = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x).squeeze()

def load_model(path="models/yield_model.pt", device="cpu"):
    """
    Charge le modèle GNN pré-entraîné pour la prédiction de rendement.
    """
    model = YieldGNN(num_node_features=1)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def smiles_to_graph(smiles):
    """
    Convertit un SMILES en Data PyG (noeuds = atomes, arêtes = liaisons).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES invalide : {smiles}")

    # Extraction des caractéristiques atomiques
    node_feats = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = [[], []]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )
    # Tous les noeuds appartiennent au même graphe (batch unique)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def predict_yield(reactions, model, device="cpu"):
    """
    Prédit le rendement pour chaque paire (réactifs, produits).
    reactions : liste de tuples (reactants_smiles, products_smiles)
    Retourne une liste de rendements estimés.
    """
    results = []
    for reactant, product in reactions:
        graph = smiles_to_graph(reactant + "." + product).to(device)
        with torch.no_grad():
            y = model(graph)
        results.append(float(y.cpu()))
    return results
