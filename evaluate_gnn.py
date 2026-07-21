import pandas as pd
import torch
import torch.nn.functional as F

from rdkit import Chem

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

activity = pd.read_csv("data/activity.csv")

activity = activity.dropna(subset=["pchembl_value"])
activity = activity[["canonical_smiles", "pchembl_value"]]

graphs = []

# --------------------------------------------------
# Convert molecules to graphs
# --------------------------------------------------

for _, row in activity.iterrows():

    mol = Chem.MolFromSmiles(row["canonical_smiles"])

    if mol is None:
        continue

    x = torch.tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    edges = []

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        edges.append([a, b])
        edges.append([b, a])

    if len(edges) == 0:
        continue

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    y = torch.tensor(
        [row["pchembl_value"]],
        dtype=torch.float
    )

    graphs.append(
        Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
    )

# --------------------------------------------------
# Same split as training
# --------------------------------------------------

_, test_graphs = train_test_split(
    graphs,
    test_size=0.2,
    random_state=42
)

test_loader = DataLoader(
    test_graphs,
    batch_size=64,
    shuffle=False
)

# --------------------------------------------------
# GCN model definition
# --------------------------------------------------

class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, data):

        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))

        x = global_mean_pool(x, data.batch)

        return self.linear(x)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN().to(device)

model.load_state_dict(
    torch.load(
        "models/gnn_model.pt",
        map_location=device
    )
)

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        pred = model(batch).view(-1)

        predictions.extend(pred.cpu().numpy())
        actual.extend(batch.y.cpu().numpy())

rmse = mean_squared_error(
    actual,
    predictions
) ** 0.5

r2 = r2_score(
    actual,
    predictions
)

print("\n========== GNN Evaluation ==========")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")