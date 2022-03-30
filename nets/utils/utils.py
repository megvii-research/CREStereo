import megengine as mge
import megengine.functional as F
import numpy as np


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]

    img = F.remap(img, coords, border_mode="constant")

    if mask:
        mask = (
            (coords[:, :, :, 0:1] < 0)
            | (coords[:, :, :, 0:1] > W - 1)
            | (coords[:, :, :, 1:2] < 0)
            | (coords[:, :, :, 1:2] > H - 1)
        )
        mask = F.logical_not(mask)
        return img, mask.astype("float32")

    return img


def coords_grid(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    y_grid, x_grid = mge.tensor(y_grid, dtype="float32"), mge.tensor(
        x_grid, dtype="float32"
    )
    coords = F.stack([x_grid, y_grid], axis=0)
    coords = F.repeat(F.expand_dims(coords, axis=0), batch, axis=0)
    return coords


def manual_pad(x, pady, padx):
    if pady > 0:
        u = F.repeat(x[:, :, 0:1, :], pady, axis=2)
        d = F.repeat(x[:, :, -1:, :], pady, axis=2)
        x = F.concat([u, x, d], axis=2)
    if padx > 0:
        l = F.repeat(x[:, :, :, 0:1], padx, axis=3)
        r = F.repeat(x[:, :, :, -1:], padx, axis=3)
        x = F.concat([l, x, r], axis=3)
    return x
