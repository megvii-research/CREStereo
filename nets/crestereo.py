import megengine.module as M
import megengine.functional as F
from megengine import amp

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import AGCL

from .attention import PositionEncodingSine, LocalFeatureTransformer


class CREStereo(M.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super(CREStereo, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        # feature network and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=self.dropout
        )
        self.update_block = BasicUpdateBlock(
            hidden_dim=self.hidden_dim, cor_planes=4 * 9, mask_size=4
        )

        # loftr
        self.self_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

        # adaptive search
        self.search_num = 9
        self.conv_offset_16 = M.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_8 = M.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.range_16 = 1
        self.range_8 = 1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, M.BatchNorm2d):
                m.eval()

    def unfold(self, x, kernel_size, dilation=1, padding=0, stride=1):
        n, c, h, w = x.shape
        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            k1, k2 = kernel_size
        else:
            assert isinstance(kernel_size, int)
            k1 = k2 = kernel_size
        x = F.sliding_window(
            x,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        x = F.reshape(x, (n, c, -1, k1 * k2))
        x = F.transpose(x, (0, 1, 3, 2))
        x = F.reshape(x, (n, c * k1 * k2, -1))
        return x

    def convex_upsample(self, flow, mask, rate=4):
        """[H/rate, W/rate, 2] -> [H, W, 2]"""
        N, _, H, W = flow.shape
        mask = F.reshape(mask, (N, 1, 9, rate, rate, H, W))
        mask = F.softmax(mask, axis=2)

        up_flow = self.unfold(rate * flow, [3, 3], padding=1)
        up_flow = F.reshape(up_flow, (N, 2, 9, 1, 1, H, W))

        up_flow = F.sum(mask * up_flow, axis=2)
        up_flow = F.transpose(up_flow, (0, 1, 4, 2, 5, 3))
        return F.reshape(up_flow, (N, 2, rate * H, rate * W))

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = F.zeros([N, 1, H, W], dtype="float32")
        _y = F.zeros([N, 1, H, W], dtype="float32")
        zero_flow = F.concat([_x, _y], axis=1).to(fmap.device)
        return zero_flow

    def forward(self, image1, image2, iters=10, flow_init=None):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        # feature network
        with amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.astype("float32")
        fmap2 = fmap2.astype("float32")

        with amp.autocast(enabled=self.mixed_precision):

            # 1/4 -> 1/8
            # feature
            fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

            # offset
            offset_dw8 = self.conv_offset_8(fmap1_dw8)
            offset_dw8 = self.range_8 * (F.sigmoid(offset_dw8) - 0.5) * 2.0

            # context
            net, inp = F.split(fmap1, [hdim], axis=1)
            net = F.tanh(net)
            inp = F.relu(inp)
            net_dw8 = F.avg_pool2d(net, 2, stride=2)
            inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
            fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
            offset_dw16 = self.conv_offset_16(fmap1_dw16)
            offset_dw16 = self.range_16 * (F.sigmoid(offset_dw16) - 0.5) * 2.0

            # context
            net_dw16 = F.avg_pool2d(net, 4, stride=4)
            inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

            # positional encoding and self-attention
            pos_encoding_fn_small = PositionEncodingSine(
                d_model=256, max_shape=(image1.shape[2] // 16, image1.shape[3] // 16)
            )
            # 'n c h w -> n (h w) c'
            x_tmp = pos_encoding_fn_small(fmap1_dw16)
            fmap1_dw16 = F.reshape(
                F.transpose(x_tmp, (0, 2, 3, 1)),
                (x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]),
            )
            # 'n c h w -> n (h w) c'
            x_tmp = pos_encoding_fn_small(fmap2_dw16)
            fmap2_dw16 = F.reshape(
                F.transpose(x_tmp, (0, 2, 3, 1)),
                (x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]),
            )

            fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
            fmap1_dw16, fmap2_dw16 = [
                F.transpose(
                    F.reshape(x, (x.shape[0], image1.shape[2] // 16, -1, x.shape[2])),
                    (0, 3, 1, 2),
                )
                for x in [fmap1_dw16, fmap2_dw16]
            ]

        corr_fn = AGCL(fmap1, fmap2)
        corr_fn_dw8 = AGCL(fmap1_dw8, fmap2_dw8)
        corr_fn_att_dw16 = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.nn.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/16
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_fn_att_dw16(
                    flow_dw16, offset_dw16, small_patch=small_patch
                )

                with amp.autocast(enabled=self.mixed_precision):
                    net_dw16, up_mask, delta_flow = self.update_block(
                        net_dw16, inp_dw16, out_corrs, flow_dw16
                    )

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.nn.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * F.nn.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/8
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_fn_dw8(flow_dw8, offset_dw8, small_patch=small_patch)

                with amp.autocast(enabled=self.mixed_precision):
                    net_dw8, up_mask, delta_flow = self.update_block(
                        net_dw8, inp_dw8, out_corrs, flow_dw8
                    )

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.nn.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.nn.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)

            with amp.autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return predictions
