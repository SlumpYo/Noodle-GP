import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from .dataset import QuantizedNpzDataset

def pad_collate(batch):
    feats, ft_tab, ft_on, nt_tab, nt_on = zip(*batch)
    B = len(feats)

    # pad raw features to same W_audio_max, so it can be batched together
    C, H = feats[0].shape[:2]
    W_audio_max = max(f.shape[2] for f in feats)
    feats_batch = torch.stack([
        F.pad(f, (0, W_audio_max - f.shape[2], 0, 0, 0, 0))
        for f in feats
    ], dim=0)  # (B, C, H, W_audio_max)

    # down-sampled frame length (32 is to account for the stride of resnet)
    W_model_max = math.ceil(W_audio_max / 32)

    # pad frame targets and build mask (contains 1 for actual data, 0 for padding)
    ft_tab_b, ft_on_b, frame_mask = [], [], []
    for t, o in zip(ft_tab, ft_on):
        pad = W_model_max - t.shape[0]
        ft_tab_b.append(F.pad(t, (0,0,0,0,0, pad)))
        ft_on_b .append(F.pad(o, (0,0,0,0,0, pad)))
        frame_mask.append(torch.cat([
            torch.ones(t.shape[0],  dtype=torch.bool),
            torch.zeros(pad,      dtype=torch.bool)
        ]))
    ft_tab_b   = torch.stack(ft_tab_b,   0) 
    ft_on_b    = torch.stack(ft_on_b,    0)
    frame_mask = torch.stack(frame_mask, 0)

    # note targets need no padding
    nt_tab_b = torch.stack(nt_tab, 0)
    nt_on_b  = torch.stack(nt_on,  0)

    return feats_batch, ft_tab_b, ft_on_b, frame_mask, nt_tab_b, nt_on_b

def build_quantized_dataloader(config_path='configs/data_config_quantized.yaml'):
    cfg = yaml.safe_load(open(config_path))
    dataset = QuantizedNpzDataset(
        npz_dir     = cfg['npz_dir'],
        n_notes     = cfg['n_notes'],
        feature_keys= cfg.get('feature_keys',['cqt','log_cqt']),
        transform   = None
    )
    return DataLoader(
        dataset,
        batch_size  = cfg.get('batch_size',8),
        shuffle     = cfg.get('shuffle',True),
        num_workers = cfg.get('num_workers',4),
        pin_memory  = cfg.get('pin_memory',True),
        collate_fn  = pad_collate,
    )