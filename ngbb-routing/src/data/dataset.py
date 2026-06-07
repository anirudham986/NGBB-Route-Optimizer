"""PyTorch Geometric HeteroData dataset wrapper.

Loads processed bipartite graphs and their corresponding branching labels
for training the GNN policy.
"""

import os
import torch
from torch_geometric.data import Dataset, HeteroData


class NGBBDataset(Dataset):
    """Dataset for NGBB bipartite graphs.
    
    Loads .pt files generated during data collection containing 
    HeteroData objects with 'variable' and 'constraint' nodes.
    """

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.files = sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.files

    def download(self):
        pass

    def process(self):
        # Processing is handled by the data collection script
        pass

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.files[idx]))
        return data


def collate_fn(batch):
    """Custom collation if needed, though PyG's DataLoader usually handles HeteroData."""
    from torch_geometric.loader import DataLoader
    return next(iter(DataLoader(batch, batch_size=len(batch))))
