import yaml
import torch
from torch.utils.data import random_split, DataLoader
from src.data_loader.dataset import QuantizedNpzDataset
from src.data_loader.dataloader import pad_collate


def build_train_val_test_loaders(
    config_path: str = 'configs/data_config_quantized.yaml',
    splits: tuple = (0.8, 0.1, 0.1),
    seed: int = 42
):
    """
    Load the full QuantizedNpzDataset and split it into train/val/test subsets,
    then wrap each in a DataLoader with the same settings as build_quantized_dataloader.
    train_loader, val_loader, test_loader
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Instantiate dataset using n_notes
    dataset = QuantizedNpzDataset(
        npz_dir      = cfg['npz_dir'],
        n_notes      = cfg.get('n_notes', 64),
        feature_keys = cfg.get('feature_keys', ['cqt', 'log_cqt']),
        transform    = None
    )

    # Determine split lengths
    total_len = len(dataset)
    train_len = int(splits[0] * total_len)
    val_len   = int(splits[1] * total_len)
    test_len  = total_len - train_len - val_len

    # Perform splits
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=generator
    )

    loader_kwargs = {
        'batch_size':  cfg.get('batch_size', 8),
        'shuffle':     True,
        'num_workers': cfg.get('num_workers', 4),
        'pin_memory':  cfg.get('pin_memory', True),
        'collate_fn':  pad_collate,
    }

    # Create DataLoaders
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   **{**loader_kwargs, 'shuffle': False})
    test_loader  = DataLoader(test_ds,  **{**loader_kwargs, 'shuffle': False})

    return train_loader, val_loader, test_loader