import megengine.module as M
import megengine.functional as F


class FlowHead(M.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = M.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = M.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = M.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(M.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = M.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        # horizontal
        hx = F.concat([h, x], axis=1)
        z = F.sigmoid(self.convz1(hx))
        r = F.sigmoid(self.convr1(hx))
        q = F.tanh(self.convq1(F.concat([r * h, x], axis=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = F.concat([h, x], axis=1)
        z = F.sigmoid(self.convz2(hx))
        r = F.sigmoid(self.convr2(hx))
        q = F.tanh(self.convq2(F.concat([r * h, x], axis=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(M.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = M.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = M.Conv2d(256, 192, 3, padding=1)
        self.convf1 = M.Conv2d(2, 128, 7, padding=3)
        self.convf2 = M.Conv2d(128, 64, 3, padding=1)
        self.conv = M.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = F.concat([cor, flo], axis=1)
        out = F.relu(self.conv(cor_flo))
        return F.concat([out, flow], axis=1)


class BasicUpdateBlock(M.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = M.Sequential(
            M.Conv2d(128, 256, 3, padding=1),
            M.ReLU(),
            M.Conv2d(256, mask_size**2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = F.concat([inp, motion_features], axis=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
