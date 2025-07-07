from torch.utils.data import DataLoader
from src.data_loader.dataset import QuantizedNpzDataset
from src.data_loader.dataloader import pad_collate
import yaml
import torch

"""
Script to get the weights by calculating class counts
"""

def get_counts():
    with open("configs/data_config_quantized.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    ds = QuantizedNpzDataset(
        npz_dir      = data_cfg['npz_dir'],
        n_notes      = model_cfg['n_notes'],
        feature_keys = data_cfg['feature_keys']
    )

    loader = DataLoader(
        ds,
        batch_size = 1,
        shuffle    = False,
        collate_fn = pad_collate,
        num_workers= data_cfg.get("num_workers", 4),
        pin_memory = data_cfg.get("pin_memory", True),
    )

    # Prepare 21‐element counters
    frame_counts = torch.zeros(21, dtype=torch.long)
    note_counts  = torch.zeros(21, dtype=torch.long)

    for feats, ft_tab, ft_on, mask, nt_tab, nt_on in loader:
        # ft_tab: (1, T_model, 6, 21)
        ft_idx = ft_tab.argmax(dim=-1).reshape(-1)   # shape: (T_model*6,)
        for i in ft_idx:
            frame_counts[i.item()] += 1

        # nt_tab: (1, N_notes, 6, 21)
        nt_idx = nt_tab.argmax(dim=-1).reshape(-1)   # shape: (N_notes*6,)
        for i in nt_idx:
            note_counts[i.item()] += 1

    return frame_counts.numpy(), note_counts.numpy()