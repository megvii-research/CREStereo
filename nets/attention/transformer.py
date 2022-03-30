"""
Transformer module proposed in "LoFTR: Detector-Free Local Feature Matching with Transformers"
Modified from: https://github.com/zju3dv/LoFTR/tree/master/src/loftr
"""

import copy
import megengine.module as M
import megengine.functional as F
from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(M.Module):
    def __init__(self, d_model, nhead, attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = M.Linear(d_model, d_model, bias=False)
        self.k_proj = M.Linear(d_model, d_model, bias=False)
        self.v_proj = M.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = M.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = M.Sequential(
            M.Linear(d_model * 2, d_model * 2, bias=False),
            M.ReLU(),
            M.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = M.LayerNorm(d_model)
        self.norm2 = M.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        query = F.reshape(
            self.q_proj(query), (bs, -1, self.nhead, self.dim)
        )  # [N, L, (H, D)] (H=8, D=256//8)
        key = F.reshape(
            self.k_proj(key), (bs, -1, self.nhead, self.dim)
        )  # [N, S, (H, D)]
        value = F.reshape(self.v_proj(value), (bs, -1, self.nhead, self.dim))
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(
            F.reshape(message, (bs, -1, self.nhead * self.dim))
        )  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(F.concat([x, message], axis=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(M.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model, nhead, layer_names, attention):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = [
            copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))
        ]
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                M.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert (
            self.d_model == feat0.shape[2]
        ), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
