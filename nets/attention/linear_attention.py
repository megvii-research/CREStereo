"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import numpy as np
import megengine.module as M
import megengine.functional as F


def elu(x, alpha=1.0):
    return F.maximum(0, x) + F.minimum(0, alpha * (F.exp(x) - 1))


def elu_feature_map(x):
    return elu(x) + 1


class LinearAttention(M.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * F.expand_dims(q_mask, (2, 3))  # [:, :, None, None]
        if kv_mask is not None:
            K = K * F.expand_dims(kv_mask, (2, 3))  # [:, :, None, None]
            values = values * F.expand_dims(kv_mask, (2, 3))  # [:, :, None, None]

        v_length = values.shape[1]
        values = values / v_length  # prevent fp16 overflow
        KV = F.sum(F.expand_dims(K, -1) * F.expand_dims(values, 3), axis=1)
        Z = 1 / (F.sum(Q * F.sum(K, axis=1, keepdims=True), axis=-1) + self.eps)
        queried_values = (
            F.sum(
                F.expand_dims(Q, -1) * F.expand_dims(KV, 1) * F.expand_dims(Z, (3, 4)),
                axis=3,
            )
            * v_length
        )

        return queried_values


class FullAttention(M.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = M.Dropout(drop_prob=attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = F.sum(F.expand_dims(queries, 2) * F.expand_dims(keys, 1), axis=-1)
        if kv_mask is not None:
            assert q_mask.dtype == np.bool_
            assert kv_mask.dtype == np.bool_
            QK[
                ~(F.expand_dims(q_mask, (2, 3)) & F.expand_dims(kv_mask, (1, 3)))
            ] = float("-inf")

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.shape[3] ** 0.5  # sqrt(D)
        A = F.softmax(softmax_temp * QK, axis=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = F.sum(F.expand_dims(A, -1) * F.expand_dims(values, 1), axis=2)

        return queried_values
