import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Optional, Sequence

class QuantizedNpzDataset(Dataset):
    """
    Returns:
    feat_tensor:  (C, H, W_audio)
    frame_tab:    (W_model, 6, 21)
    frame_onset:  (W_model, 6, 21)
    note_tab:     (n_notes, 6, 21)
    note_onset:   (n_notes, 6, 21)
    """
    def __init__(
        self,
        npz_dir: str,
        n_notes: int,
        feature_keys=('cqt','log_cqt'),
        transform=None,
        file_list: Optional[Sequence[str]] = None,
    ):
        # list override for 6-fold validation based on guitar player
        if file_list is not None:
            paths = []
            for p in file_list:
                # try file as-is
                if os.path.isfile(p):
                    paths.append(p)
                # try joining to npz_dir
                elif os.path.isfile(os.path.join(npz_dir, p)):
                    paths.append(os.path.join(npz_dir, p))
                else:
                    raise FileNotFoundError(f"Cannot find file in file_list: '{p}'")
            self.paths = sorted(paths)
        else:
            self.paths = sorted(
                os.path.join(npz_dir, f)
                for f in os.listdir(npz_dir)
                if f.endswith('.npz')
            )

        self.feature_keys = feature_keys
        self.n_notes      = n_notes
        self.transform    = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        archive = np.load(self.paths[idx])

        # input features (C, H, W_audio) —
        feats = [archive[k] for k in self.feature_keys]
        feats = [f.T if f.ndim == 2 else f for f in feats]
        feat_tensor = torch.from_numpy(np.stack(feats, 0)).float()

        # raw frame gt at audio rate (T_audio, 6, 21)
        raw_ft   = torch.from_numpy(archive['frame_tab']).float()
        raw_on   = torch.from_numpy(archive['frame_tab_onset']).float()

        W_audio = feat_tensor.shape[2]
        W_model = math.ceil(W_audio / 32)

        # flatten channels (1,126,T_audio)
        ft = raw_ft.permute(1,2,0).reshape(1, 6*21, -1)
        on = raw_on.permute(1,2,0).reshape(1, 6*21, -1)

        ft_ds = F.interpolate(ft, size=W_model, mode='nearest') # downsampled data, using nearest
        on_ds = F.interpolate(on, size=W_model, mode='nearest') # must be the same as in infer script

        frame_tab   = ft_ds.reshape(6,21,W_model).permute(2,0,1)
        frame_onset = on_ds.reshape(6,21,W_model).permute(2,0,1)

        # gt for notes
        raw_nt    = torch.from_numpy(archive['tab']).float()
        raw_nt_on = torch.from_numpy(archive['tab_onset']).float()
        if raw_nt.shape[0] != self.n_notes: # if n_notes is variable
            nt   = raw_nt.permute(1,2,0).reshape(1,6*21,-1)
            non  = raw_nt_on.permute(1,2,0).reshape(1,6*21,-1)
            nt_ds  = F.interpolate(nt,  size=self.n_notes, mode='nearest')
            non_ds = F.interpolate(non, size=self.n_notes, mode='nearest')
            note_tab    = nt_ds.reshape(6,21,self.n_notes).permute(2,0,1)
            note_onset  = non_ds.reshape(6,21,self.n_notes).permute(2,0,1)
        else: # in our case just raw
            note_tab    = raw_nt
            note_onset  = raw_nt_on

        # online augmentation (unused)
        if self.transform:
            feat_tensor = self.transform(feat_tensor)

        return feat_tensor, frame_tab, frame_onset, note_tab, note_onset