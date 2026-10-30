import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    """4-layer CNN that predicts the normalised residual.

    input:  (B, 20, 97, 241)  - two-stage reconstructed fields (normalised)
    output: (B, 20, 97, 241)  - predicted residual (normalised)
    final  = reconstruction + output * std   (done outside the model)
    """

    def __init__(self, in_channels=20, mid_channels=64, n_layers=4, kernel_size=3,
                 dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1] * n_layers
        assert len(dilations) == n_layers, f"need {n_layers} dilations, got {len(dilations)}"
        layers = []
        for i in range(n_layers):
            ch_in = in_channels if i == 0 else mid_channels
            ch_out = in_channels if i == n_layers - 1 else mid_channels
            dilation = dilations[i]
            padding = dilation * (kernel_size - 1) // 2
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding,
                                    dilation=dilation))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)


class ResBlockCNN(nn.Module):
    """4-conv CNN with one residual block, no dilation.

    control for isolating skip-connection vs dilation effect:
        stem: Conv2d 20->64, k3, dil=1, ReLU
        res:  Conv2d 64->64, k3, dil=1, ReLU
              Conv2d 64->64, k3, dil=1, (+) skip, ReLU
        head: Conv2d 64->20, k3, dil=1, no activation
    """

    def __init__(self, in_channels=20, mid_channels=64, kernel_size=3, block_dilations=None):
        super().__init__()
        if block_dilations is None:
            block_dilations = [1, 1]
        assert len(block_dilations) == 2, f"need 2 block dilations, got {len(block_dilations)}"
        d1, d2 = block_dilations
        pad = lambda d: d * (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=pad(1)),
            nn.ReLU(inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=pad(d1), dilation=d1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=pad(d2), dilation=d2),
        )
        self.res_act = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(mid_channels, in_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        h = self.stem(x)
        h = self.res_act(h + self.res_block(h))
        return self.head(h)


class MultiScaleCNN(nn.Module):
    """Parallel multi-scale branches (dil=1/2/4) fused with 1x1 + two 3x3 convs.

    input:  (B, 20, 97, 241)
    output: (B, 20, 97, 241)  - normalised residual
    """

    def __init__(self, in_channels=20, mid_channels=64, kernel_size=3, branch_channels=32,
                 dilations=(1, 2, 4)):
        super().__init__()
        pad = lambda d: d * (kernel_size - 1) // 2
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size, padding=pad(d),
                          dilation=d),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        n_branches = len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * n_branches, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        h = torch.cat(feats, dim=1)
        return self.fusion(h)


class UNet2(nn.Module):
    """Encoder-decoder U-Net with configurable depth and base width.

    input:  (B, 20, 97, 241)
    output: (B, 20, 97, 241)  - normalised residual (no final activation)

    level channels:  mid_channels * 2^i  for i in 0..depth

    depth=2, base=32:
        enc1: 20 -> 32 -> 32     down1: MaxPool2d -> 32 -> 64
        enc2: 64 -> 64 -> 64     down2: MaxPool2d -> 64 -> 128
        bottleneck: 128 -> 128 -> 128
        up2: 128 -> 64, concat enc2 skip, 128 -> 64 -> 64
        up1: 64 -> 32, concat enc1 skip, 64 -> 32 -> 32
        head: 32 -> 20 (no activation)

    depth=3, base=32:
        97x241 -> 48x120 -> 24x60 -> 12x30 -> 24x60 -> 48x120 -> 97x241
        channels 20 -> 32 -> 64 -> 128 -> 256 (bottleneck)

    bottleneck_channels: override the bottleneck width. Default (None) is
    mid_channels * 2^depth. Set e.g. 128 to reproduce the original
    depth-2 c64 architecture (bottleneck 128, not 256).

    Downsampling uses MaxPool2d(2) (floor-divide), so on odd sizes like
    97x241 the decoder upsampling may not perfectly restore the input size.
    We therefore use F.interpolate(..., size=skip.shape[-2:]) instead of
    scale_factor so decoder features always match the skip feature shape
    before concat.
    """

    def __init__(self, in_channels=20, mid_channels=32, kernel_size=3, depth=2,
                 bottleneck_channels=None):
        super().__init__()
        pad = kernel_size // 2
        chs = [mid_channels * (2 ** i) for i in range(depth)]
        if bottleneck_channels is None:
            bottleneck_channels = chs[-1] * 2

        def double_conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, padding=pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size, padding=pad),
                nn.ReLU(inplace=True),
            )

        def down_block(cin, cout):
            return nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(cin, cout, kernel_size, padding=pad),
            )

        self.encs = nn.ModuleList([double_conv(in_channels, chs[0])])
        for i in range(1, depth):
            self.encs.append(double_conv(chs[i], chs[i]))
        self.downs = nn.ModuleList(
            [down_block(chs[i], chs[i + 1]) for i in range(depth - 1)]
        )
        self.downs.append(down_block(chs[-1], bottleneck_channels))
        self.bottleneck = double_conv(bottleneck_channels, bottleneck_channels)

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.ups.append(nn.Conv2d(bottleneck_channels, chs[-1], kernel_size, padding=pad))
        self.decs.append(double_conv(chs[-1] * 2, chs[-1]))
        for i in range(depth - 1, 0, -1):
            self.ups.append(nn.Conv2d(chs[i], chs[i - 1], kernel_size, padding=pad))
            self.decs.append(double_conv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], in_channels, kernel_size, padding=pad)

    def forward(self, x):
        skips = []
        h = x
        for i, enc in enumerate(self.encs):
            h = enc(h)
            skips.append(h)
            h = self.downs[i](h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            h = up(h)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))
        return self.head(h)


class SpectralConv2d(nn.Module):
    """Fourier-domain linear operator: rfft2 -> truncate to (modes1, modes2)
    -> complex matmul over the channel axis -> irfft2.

    Weights are stored as real tensors with a trailing size-2 axis and viewed as
    complex, so optimiser state and checkpoints stay plain float32.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # two blocks: the positive and negative halves of the first frequency axis
        self.w_pos = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.w_neg = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def _cmul(x, w):
        # x (B, Cin, m1, m2) complex ; w (Cin, Cout, m1, m2) complex
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        b, c, h, w = x.shape
        # torch.fft has no fp16 kernel; keep this branch in float32 under --amp
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            xf = torch.fft.rfft2(x, norm="ortho")          # (B, C, h, w//2+1)
            n_freq = xf.shape[-1]
            m1 = min(self.modes1, h // 2)
            m2 = min(self.modes2, n_freq)
            out = torch.zeros(b, self.out_channels, h, n_freq,
                              dtype=torch.cfloat, device=x.device)
            wp = torch.view_as_complex(self.w_pos)[:, :, :m1, :m2]
            wn = torch.view_as_complex(self.w_neg)[:, :, :m1, :m2]
            out[:, :, :m1, :m2] = self._cmul(xf[:, :, :m1, :m2], wp)
            out[:, :, -m1:, :m2] = self._cmul(xf[:, :, -m1:, :m2], wn)
            return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    """Fourier Neural Operator over the CONUS window.

    input:  (B, 20, 97, 241)  - two-stage reconstructed fields (normalised)
    output: (B, 20, 97, 241)  - predicted residual (normalised)

    lift 1x1 -> [spectral conv + pointwise bypass + GroupNorm + GELU] x n_layers
    -> project 1x1 -> GELU -> 1x1.

    Parameters are dominated by the spectral weights, 4 * width^2 * m1 * m2 per
    layer, so width and mode count trade against each other at a fixed budget.
    That trade is the thing worth measuring: modes control what scales the
    operator can act on, width controls how many channel interactions it can
    represent, and it is not obvious which one the PCA residual needs.
    """

    def __init__(self, in_channels=20, width=24, modes1=18, modes2=18,
                 n_layers=4, project_channels=128, use_grid=True):
        super().__init__()
        self.use_grid = use_grid
        self.lift = nn.Conv2d(in_channels + (2 if use_grid else 0), width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.bypass = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(1, width) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, project_channels, 1),
            nn.GELU(),
            nn.Conv2d(project_channels, in_channels, 1),
        )

    def _grid(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx]).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, x):
        if self.use_grid:
            x = torch.cat([x, self._grid(x)], dim=1)
        h = self.lift(x)
        for spec, byp, norm in zip(self.spectral, self.bypass, self.norms):
            h = F.gelu(norm(spec(h).to(h.dtype) + byp(h)))
        return self.project(h)
