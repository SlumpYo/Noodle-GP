import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.models.resnet18transformer2tab import ResNetTransformerFull as NoodleGP18
from src.models.resnet34transformer2tab import ResNetTransformerFull as NoodleGP34
from src.data_loader.dataset import QuantizedNpzDataset
from src.data_loader.dataloader import pad_collate

base_out_dir = "nearest_results"

def tab2pitch(tab: np.ndarray) -> np.ndarray:
    """
    Convert (T,6,21) one-hot tablature to (T,44) pitch one-hot.
    Used in TDR.
    """
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    T = tab.shape[0]
    pitch = np.zeros((T, 44), dtype=int)
    for t in range(T):
        for s in range(6):
            fret = tab[t, s].argmax()
            if fret < 20:
                pitch[t, rel_string_pitches[s] + fret] = 1
    return pitch

def calculate_tdr(tab_pred: np.ndarray, tab_gt: np.ndarray) -> float:
    pred_tab = tab_pred[:, :, :-1]
    gt_tab   = tab_gt  [:, :, :-1]
    TP_tab   = (pred_tab & gt_tab).sum() # Get only the instances where tab matches
    pred_pitch = tab2pitch(tab_pred)
    gt_pitch   = tab2pitch(tab_gt)
    TP_pitch = (pred_pitch & gt_pitch).sum() # Get only the instances where pitch matches
    return float(TP_tab) / float(TP_pitch) if TP_pitch > 0 else 0.0

def evaluate_fold(fold_id: int, cfg: dict, device: torch.device, base_folder: str, file_list: list) -> dict:
    # Choose model (depending on resnet)
    ModelClass = NoodleGP34 if "34" in base_folder else NoodleGP18
    model = ModelClass(
        in_channels=len(cfg['feature_keys']),
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'],
        dim_feedforward=cfg['dim_feedforward'],
        dropout=cfg['dropout'],
        n_notes=cfg['n_notes']
    ).to(device)
    ckpt_path = os.path.join(base_folder, f"fold{fold_id}", f"best_fold{fold_id}.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k: v for k, v in state.items()})
    model.eval()

    # Instantiate dataloader with custom list (unaugmented files, just cqts)
    dataset = QuantizedNpzDataset(
        npz_dir=None,            # unused when file_list provided
        n_notes=cfg['n_notes'],
        feature_keys=cfg['feature_keys'], # cqt and loq(cqt)
        transform=None,
        file_list=file_list
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # batch size 1
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=pad_collate # not really necessary
    )

    # accumulators for logging, per-file and globally
    sums = {k: 0.0 for k in [
        'frame_p','frame_r','frame_f',
        'note_p','note_r','note_f',
        'frame_tdr','note_tdr'
    ]}
    n_files = 0
    global_frame_gt, global_frame_pr = [], []
    global_note_gt,  global_note_pr  = [], []
    all_gt_labels,   all_pr_labels   = [], []

    # infer loop, 
    for feats, ft_tab, ft_on, frame_mask, nt_tab, nt_on in tqdm(loader, desc=f"Fold {fold_id}"):
        n_files += 1
        feats = feats.to(device)
        
        full_gt_tab = ft_tab.squeeze(0).cpu().numpy()    # (T_model, 6, 21)
        full_gt_note = nt_tab.squeeze(0).cpu().numpy()  # (n_notes, 6, 21)
        
        # Convert to full integers for metrics
        full_gt_tab_int = full_gt_tab.astype(np.int32)
        full_gt_note_int = full_gt_note.astype(np.int32)

        # get predictions
        with torch.no_grad():
            f_logits, n_logits = model(feats)
        f_logits = f_logits.squeeze(1).cpu().numpy()  # (Tm,6,21)
        n_logits = n_logits.squeeze(1).cpu().numpy()  # (n_notes,6,21)

        # one-hot decode to get discrete data for measuring
        frame_pr = np.zeros_like(f_logits, dtype=int)
        note_pr  = np.zeros_like(n_logits, dtype=int)
        for t in range(f_logits.shape[0]):
            for s in range(6):
                frame_pr[t, s, f_logits[t, s].argmax()] = 1
        for n in range(n_logits.shape[0]):
            for s in range(6):
                note_pr[n, s, n_logits[n, s].argmax()] = 1

        # compute p r and f, removing no-play
        fp  = frame_pr[:, :, :-1].reshape(-1)
        fg  = full_gt_tab[:, :, :-1].reshape(-1)
        np2 = note_pr[:, :, :-1].reshape(-1)
        ng  = full_gt_note[:, :, :-1].reshape(-1)

        # per-file metrics
        p_f, r_f, f_f, _ = precision_recall_fscore_support(fg, fp, average='binary', zero_division=0)
        p_n, r_n, f_n, _ = precision_recall_fscore_support(ng, np2, average='binary', zero_division=0)
        for key, val in zip(
            ['frame_p','frame_r','frame_f','note_p','note_r','note_f'],
            [p_f, r_f, f_f, p_n, r_n, f_n]
        ):
            sums[key] += val

        # global metrics
        global_frame_gt.extend(fg.tolist())
        global_frame_pr.extend(fp.tolist())
        global_note_gt.extend(ng.tolist())
        global_note_pr.extend(np2.tolist())

        # TDR
        sums['frame_tdr'] += calculate_tdr(frame_pr, full_gt_tab_int)
        sums['note_tdr']  += calculate_tdr(note_pr, full_gt_note_int)

        # Confusion data
        T, S, _ = full_gt_tab_int.shape
        for t in range(T):
            for s in range(S):
                g = full_gt_tab_int[t,s].argmax() # collapse to only true value
                p = frame_pr[t,s].argmax()
                if g < 20:
                    all_gt_labels.append(s*20 + g)
                    all_pr_labels.append(s*20 + p)

    # per-file averages 
    for k in sums:
        sums[k] /= n_files
    out = {'fold': fold_id, **sums}

    # Global averages
    pg, rg, fg, _ = precision_recall_fscore_support(global_frame_gt, global_frame_pr, average='binary', zero_division=0)
    pn, rn, fn, _ = precision_recall_fscore_support(global_note_gt,  global_note_pr,  average='binary', zero_division=0)
    out.update({
        'global_frame_p': pg,
        'global_frame_r': rg,
        'global_frame_f': fg,
        'global_note_p' : pn,
        'global_note_r' : rn,
        'global_note_f' : fn
    })

    # Confusion + per-class p r and f
    labels = np.arange(120)
    cm     = confusion_matrix(all_gt_labels, all_pr_labels, labels=labels)
    prf    = precision_recall_fscore_support(all_gt_labels, all_pr_labels, labels=labels, zero_division=0)

    result_dir = os.path.join(base_out_dir, f"result_{base_folder.replace('/','_')}")
    os.makedirs(os.path.join(result_dir,'fold_confmat'), exist_ok=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(result_dir,'fold_confmat',f"confmat_{fold_id}.csv")
    )
    pd.DataFrame({'precision':prf[0],'recall':prf[1],'f1':prf[2]}, index=labels).to_csv(
        os.path.join(result_dir,'fold_confmat',f"class_prf_{fold_id}.csv")
    )
    # counts
    pred_counts    = cm.sum(axis=0)
    correct_counts = np.diag(cm)
    with open(os.path.join(result_dir,'fold_confmat',f"pred_counts_{fold_id}.txt"),'w') as f:
        f.write("class_id\tpredicted_count\n")
        for cls,cnt in zip(labels,pred_counts): f.write(f"{cls}\t{cnt}\n")
    with open(os.path.join(result_dir,'fold_confmat',f"correct_counts_{fold_id}.txt"),'w') as f:
        f.write("class_id\tcorrect_count\n")
        for cls,cnt in zip(labels,correct_counts): f.write(f"{cls}\t{cnt}\n")

    return out


def main():
    # load configs
    models = ["evalModels/modelSmallNoAugment2", "evalModels/model"]
    os.makedirs(base_out_dir, exist_ok=True)
    for base in models:
        print(f"Evaluating {base}")
        cfg_path = "configs/modelSmall.yaml" if "Small" in base else "configs/model_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # collect test files, no-aug originals
        all_files = sorted(glob.glob("data/processed/npz/original/split/*.npz"))
        results = []
        for fold in range(6):
            results.append(evaluate_fold(fold, cfg, device, base, all_files))
        out_dir = os.path.join(base_out_dir, f"result_{base.replace('/','_')}")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(
            os.path.join(out_dir, 'metrics_per_fold.csv'), index=False, float_format='%.4f'
        )
        print("Done.")

if __name__ == '__main__':
    main()