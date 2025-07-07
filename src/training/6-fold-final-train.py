"""
Full training script for ResNetX-Transformer tablature model with
6-fold cross-validation (leave-one-player-out),
Usage (if you want to override cfg):
    python -m src.training.train \
        [--epochs N] [--lr LR] [--batch-size BS] \
        [--device DEVICE] [--output-dir DIR] \
        [--patience P] [--min-delta D]
"""
import os
import math
import argparse
import yaml
import logging
import glob
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from src.data_loader.dataset import QuantizedNpzDataset
from src.data_loader.dataloader import pad_collate
from src.models.resnet18transformer2tab import ResNetTransformerFull
from src.data_loader.data_data import get_counts


def parse_args():
    p = argparse.ArgumentParser("Train dual-headed ResNet-Transformer with 6-fold CV")
    p.add_argument("config",     type=str)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str,   default=None)
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--min-delta",  type=float, default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--weight-decay",      type=float, default=None)
    p.add_argument("--cross-entropy-weight", type=int, default=None)
    p.add_argument("--seed",       type=int,   default=None)
    return p.parse_args()


class Trainer:
    def __init__(self, args, data_cfg, model_cfg, train_files, val_files, test_files, fold_idx):
        self.fold_idx = fold_idx
        # Hyperparameters and device
        self.epochs               = args.epochs or model_cfg.get("epochs", 125)
        self.lr                   = args.lr or float(model_cfg.get("learning_rate", 1e-4))
        self.batch_size           = args.batch_size or data_cfg.get("batch_size", 32)
        self.device               = torch.device(args.device)
        base_output               = args.output_dir or model_cfg.get("output_dir", "checkpoints")
        self.output_dir           = os.path.join(base_output, f"fold{fold_idx}")
        self.dropout              = args.dropout or model_cfg.get("dropout", 0.1)
        self.weight_decay         = args.weight_decay or model_cfg.get("weight_decay", 0)
        self.cross_entropy_weight = args.cross_entropy_weight or model_cfg.get("cross_entropy_weight", 2)
        self.seed                 = args.seed or model_cfg.get("seed", 42)
        os.makedirs(self.output_dir, exist_ok=True)

        # Early stopping and best F1 (for patience)
        self.patience   = args.patience or model_cfg.get("patience", 10)
        self.min_delta  = args.min_delta or model_cfg.get("early_stopping_min_delta", 1e-4)
        self.no_improve = 0
        self.best_f1    = 0.0

        # Logger
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"fold{self.fold_idx}_run_{datetime.now():%Y%m%d_%H%M%S}.log"))
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        self.logger = logging.getLogger(f"Trainer.fold{self.fold_idx}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh); self.logger.addHandler(ch)

        # Datasets and DataLoaders
        train_ds = QuantizedNpzDataset(
            npz_dir   = data_cfg['npz_dir'],
            file_list = train_files,
            n_notes      = model_cfg.get("n_notes", 64),
            feature_keys = data_cfg.get("feature_keys", ['cqt', 'log_cqt']),
        )
        val_ds = QuantizedNpzDataset(
            npz_dir   = data_cfg['npz_dir'],
            file_list = val_files,
            n_notes      = model_cfg.get("n_notes", 64),
            feature_keys = data_cfg.get("feature_keys", ['cqt', 'log_cqt']),
        )
        test_ds = QuantizedNpzDataset(
            npz_dir   = data_cfg['npz_dir'],
            file_list = test_files,
            n_notes      = model_cfg.get("n_notes", 64),
            feature_keys = data_cfg.get("feature_keys", ['cqt', 'log_cqt']),
        )
        loader_kwargs = dict(
            batch_size  = self.batch_size,
            num_workers = data_cfg.get("num_workers", 4),
            pin_memory  = data_cfg.get("pin_memory", True),
            collate_fn  = pad_collate
        )
        self.train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
        self.val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
        self.test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

        # Model initialization
        in_ch = len(data_cfg.get("feature_keys", ['cqt', 'log_cqt']))
        self.model = ResNetTransformerFull(
            in_channels         = in_ch,
            d_model             = model_cfg.get("d_model", 512),
            nhead               = model_cfg.get("nhead", 8),
            num_encoder_layers  = model_cfg.get("num_encoder_layers", 1),
            num_decoder_layers  = model_cfg.get("num_decoder_layers", 4),
            dim_feedforward     = model_cfg.get("dim_feedforward", 2048),
            dropout             = self.dropout,
            n_notes             = model_cfg.get("n_notes", 64)
        ).to(self.device)

        # Freeze ResNet.layer4
        for p in self.model.backbone.resnet.layer4.parameters(): p.requires_grad = False

        # Loss weights (if cew == 1 or 2, else just all ones)
        if self.cross_entropy_weight in (1,2):
            frame_counts, note_counts = get_counts()
            tot_frames = frame_counts.sum(); tot_notes = note_counts.sum()
            print(f"frame_counts: {frame_counts}")
            print(f"note: {note_counts}")
            frame_w = (tot_frames/(21.0*frame_counts)).astype(np.float32)
            frame_w = np.nan_to_num(frame_w, nan=0.1, posinf=600, neginf=0.1)
            note_w  = (tot_notes/(21.0*note_counts)).astype(np.float32)
            if self.cross_entropy_weight == 2:
                frame_w = np.clip(frame_w, 0.1, 600.0)
                note_w  = np.clip(note_w, 0.1, 600.0)
            self.frame_weight = torch.from_numpy(frame_w).to(self.device)
            self.note_weight  = torch.from_numpy(note_w).to(self.device)
        else:
            self.frame_weight = torch.ones(21, device=self.device)
            self.note_weight  = torch.ones(21, device=self.device)

        self.frame_ce = nn.CrossEntropyLoss(weight=self.frame_weight, ignore_index=-100).to(self.device)
        self.note_ce  = nn.CrossEntropyLoss(weight=self.note_weight,  ignore_index=-100).to(self.device)

        # Optimizer and scheduler
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        total_steps  = self.epochs * len(self.train_loader)
        warmup_steps = int(0.1 * total_steps)
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingWarmRestarts(self.optimizer, T_0=len(self.train_loader)*10, T_mult=2, eta_min=self.lr*0.01)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])

    # epoch 
    def train_epoch(self):
        self.model.train(); running=0.0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        IGN= -100
        for feats, ft_tab, ft_on, mask, nt_tab, nt_on in pbar:
            feats, ft_tab, ft_on, mask, nt_tab, nt_on = [x.to(self.device) for x in (feats, ft_tab, ft_on, mask, nt_tab, nt_on)]
            B,T,S,C = *mask.shape, 6, 21
            f_logits, n_logits = self.model(feats)
            # frame loss
            f = f_logits.permute(1,0,2,3).reshape(-1,C)
            mask_f = mask.unsqueeze(-1).expand(-1,-1,S).reshape(-1)
            labels_f = ft_tab.reshape(-1,C).argmax(-1)
            loss_f = self.frame_ce(f[mask_f], labels_f[mask_f])
            # note loss
            n = n_logits.permute(1,0,2,3).reshape(-1,C)
            labels_n = nt_tab.reshape(-1,C).argmax(-1)
            loss_n = self.note_ce(n, labels_n)
            loss = loss_f + loss_n
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(),1.0)
            self.optimizer.step()
            self.scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.4f}")
        return running/len(self.train_loader)

    # Validation is run purely on unaugmented data
    def validate(self):
        self.model.eval(); val_loss=0.0
        all_fp,all_fg,all_np,all_ng=[],[],[],[]
        with torch.no_grad():
            for feats, ft_tab, ft_on, mask, nt_tab, nt_on in self.val_loader:
                feats, ft_tab, ft_on, mask, nt_tab, nt_on = [x.to(self.device) for x in (feats, ft_tab, ft_on, mask, nt_tab, nt_on)]
                f_logits,n_logits = self.model(feats)
                # frame
                B,T,S,C = *mask.shape,6,21
                f = f_logits.permute(1,0,2,3).reshape(B, T*S, C)
                mask_f = mask.unsqueeze(-1).expand(-1,-1,S).reshape(B,T*S)
                labels_f = ft_tab.reshape(B,T*S,C).argmax(-1)
                loss_f = self.frame_ce(f.reshape(-1,C)[mask_f.reshape(-1)], labels_f.reshape(-1)[mask_f.reshape(-1)])
                # onset as frame loss 2
                labels_on=ft_on.reshape(B,T*S,C).argmax(-1)
                loss_on = self.frame_ce(f.reshape(-1,C)[mask_f.reshape(-1)], labels_on.reshape(-1)[mask_f.reshape(-1)])
                # note
                n = n_logits.permute(1,0,2,3).reshape(-1,C)
                labels_n=nt_tab.reshape(-1,C).argmax(-1)
                loss_n=self.note_ce(n, labels_n)
                labels_non=nt_on.reshape(-1,C).argmax(-1)
                loss_non=self.note_ce(n, labels_non)
                val_loss += (loss_f+loss_on+loss_n+loss_non).item()
                # F1 frame
                pred_f = f_logits.permute(1,0,2,3).argmax(-1)  # (B,T,6)
                gt_f   = ft_tab.argmax(-1)
                valid_mask = mask
                pf = pred_f[valid_mask].reshape(-1).cpu().numpy()
                gf = gt_f[valid_mask].reshape(-1).cpu().numpy()
                keep = gf<20
                all_fp.append(pf[keep]); all_fg.append(gf[keep])
                # F1 note
                pred_n = n_logits.permute(1,0,2,3).argmax(-1).reshape(-1).cpu().numpy()
                gn = nt_tab.argmax(-1).reshape(-1).cpu().numpy()
                k2 = gn<20
                all_np.append(pred_n[k2]); all_ng.append(gn[k2])
        # averaging mode set to macro, as guitarset is very unbalanced
        frame_f1 = f1_score(np.concatenate(all_fg), np.concatenate(all_fp), average='macro')
        note_f1  = f1_score(np.concatenate(all_ng), np.concatenate(all_np), average='macro')
        return val_loss/len(self.val_loader), frame_f1, note_f1

    def run(self):
        for epoch in range(1, self.epochs+1):
            if epoch==1:
                for p in self.model.backbone.resnet.layer4.parameters(): p.requires_grad=True
                self.optimizer.add_param_group({'params': self.model.backbone.resnet.layer4.parameters(),'lr':self.lr})
                self.logger.info("→ Unfroze and fine-tuning ResNet.layer4")
            tr_loss = self.train_epoch()
            val_loss, f1f, f1n = self.validate()
            lr = self.scheduler.get_last_lr()[0]
            total_f1 = f1f+f1n
            self.logger.info(f"[{epoch}/{self.epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} frame_f1={f1f:.4f} note_f1={f1n:.4f} lr={lr:.2e}")
            if total_f1>self.best_f1+self.min_delta:
                self.best_f1=total_f1; self.no_improve=0
                fname=f"best_fold{self.fold_idx}.pt"
                torch.save(self.model.state_dict(),os.path.join(self.output_dir,fname))
                self.logger.info("→ New best model saved")
            else:
                self.no_improve+=1
                if self.no_improve>=self.patience:
                    self.logger.info("→ Early stopping triggered.")
                    break
        self.logger.info(f"Fold complete: best_f1={self.best_f1:.4f}")


def main():
    args = parse_args()

    with open("configs/data_config_quantized.yaml") as f:
        data_cfg = yaml.safe_load(f)

    cfg_path = os.path.join("configs", "test_runs", f"{args.config}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    with open(cfg_path) as f:
        model_cfg = yaml.safe_load(f)

    if "npz_dir" not in model_cfg:
        raise KeyError(f"'npz_dir' must be set in {cfg_path}")
    data_cfg["npz_dir"] = model_cfg["npz_dir"]

    all_files = glob.glob(os.path.join(data_cfg['npz_dir'], "*.npz"))

    for fold in range(6):
        random.seed(model_cfg.get("seed", 42))
        test_files = [f for f in all_files if os.path.basename(f).startswith(f"0{fold}_")]
        print("test: ", len(test_files))
        dev_files  = [f for f in all_files if not os.path.basename(f).startswith(f"0{fold}_")]
        print("dev: ", len(dev_files))
        random.shuffle(dev_files)
        cut        = int(round(0.9 * len(dev_files)))
        train_files = dev_files[:cut]
        # remove augmented files from validate
        val_candidates = dev_files[cut:]
        val_files      = [f for f in val_candidates if not os.path.basename(f).endswith('_aug.npz')]

        trainer = Trainer(
            args,
            data_cfg,
            model_cfg,
            train_files,
            val_files,
            test_files,
            fold
        )
        trainer.run()


if __name__ == "__main__":
    main()