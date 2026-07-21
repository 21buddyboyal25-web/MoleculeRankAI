import joblib
import pandas as pd
import torch
import torch.nn.functional as F

from rdkit import Chem

from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --------------------------------------------------
# Load activity dataset
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

print("Graphs created:", len(graphs))

# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

train_graphs, test_graphs = train_test_split(
    graphs,
    test_size=0.2,
    random_state=42
)

train_loader = DataLoader(
    train_graphs,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_graphs,
    batch_size=64
)

# --------------------------------------------------
# Graph Neural Network
# --------------------------------------------------

class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)

        self.linear = torch.nn.Linear(64, 1)

    def forward(self, data):

        x = self.conv1(
            data.x,
            data.edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            data.edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            data.batch
        )

        return self.linear(x)

# --------------------------------------------------
# Train
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GCN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = torch.nn.MSELoss()

print("Training GNN...")

for epoch in range(20):

    model.train()

    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch).view(-1)

        loss = loss_fn(
            pred,
            batch.y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1:02d} | Loss = {total_loss:.4f}"
    )

# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "models/gnn_model.pt"
)

print("Saved models/gnn_model.pt")