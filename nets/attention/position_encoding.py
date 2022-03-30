import math
import megengine.module as M
import megengine.functional as F


class PositionEncodingSine(M.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = F.zeros((d_model, *max_shape))
        y_position = F.expand_dims(F.cumsum(F.ones(max_shape), 0), 0)
        x_position = F.expand_dims(F.cumsum(F.ones(max_shape), 1), 0)
        div_term = F.exp(
            F.arange(0, d_model // 2, 2) * (-math.log(10000.0) / d_model // 2)
        )
        div_term = F.expand_dims(div_term, (1, 2))  # [C//4, 1, 1]
        pe[0::4, :, :] = F.sin(x_position * div_term)
        pe[1::4, :, :] = F.cos(x_position * div_term)
        pe[2::4, :, :] = F.sin(y_position * div_term)
        pe[3::4, :, :] = F.cos(y_position * div_term)

        self.pe = F.expand_dims(pe, 0)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.shape[2], : x.shape[3]].to(x.device)
