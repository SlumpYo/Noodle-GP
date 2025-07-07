import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base_model import BaseModel
from src.modules.ResNet18 import ResNet18Backbone
from src.modules.positional_encoding import PositionalEncoding
from src.modules.transformerPAK.transformer import TransformerModule

class ResNetTransformerFull(BaseModel):
    """
    ResNet18
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = False,
        n_notes: int = 64,
    ):
        super().__init__()
        # initially frozen ResNet backbone
        self.backbone = ResNet18Backbone(in_channels)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # source projection + positional encoding
        self.src_proj = nn.Linear(512, d_model)
        self.src_pos  = PositionalEncoding(d_model)

        # transformer module
        self.transformer = TransformerModule(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        # frame and note heads
        self.frame_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6 * 21)
        )
        self.note_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6 * 21)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.n_notes = n_notes

    def encode(self, feats: torch.Tensor):
        # CNN --> feature map --> (S_src, B, d_model)
        fmap = self.backbone(feats)            # (B,512,H,W)
        x    = fmap.mean(dim=2).permute(2, 0, 1)  # (W, B, 512)
        x    = self.src_proj(x)
        x    = self.src_pos(x)
        pad_mask = x.abs().sum(-1) == 0       # (W, B)
        src_kpm  = pad_mask.transpose(0,1)     # (B, W)
        return x, src_kpm

    def _decimate(self, feats: torch.Tensor):
        #(S, B, D) --> (n_notes, B, D)
        S, B, D = feats.shape
        tmp = feats.permute(1,2,0)            # (B, D, S)
        dec = F.interpolate(
            tmp,
            size=self.n_notes,
            mode='linear',
            align_corners=False
        )                                     # (B, D, n_notes)
        return dec.permute(2,0,1)             # (n_notes, B, D)

    def forward(
        self,
        feats: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_kpm:  Optional[torch.Tensor] = None,
    ):
        # Encode
        memory, src_kpm = self.encode(feats)   # (S_src, B, D)

        # frame-level predictions
        S_src, B, D = memory.shape
        f_logits = self.frame_head(memory.reshape(-1, D))
        f_logits = f_logits.view(S_src, B, 6, 21)
        frame_pred = self.softmax(f_logits)
        
        dec_out = memory

        # note-level predictions
        note_feats = self._decimate(dec_out)   # (n_notes, B, D)
        n_logits   = self.note_head(note_feats.reshape(-1, D))
        n_logits   = n_logits.reshape(self.n_notes, B, 6, 21)
        note_pred  = self.softmax(n_logits)

        return frame_pred, note_pred