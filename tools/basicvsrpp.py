"""
BasicVSR++ without mmcv/mmedit, so it can be exported.

The upstream model (`dev/BasicVSR_PlusPlus`) is an OpenMMLab package: the network
definition itself is plain PyTorch, but it imports four symbols from `mmcv` and
registers itself with `mmedit`'s registry. mmcv is not installed here and its
`ModulatedDeformConv2d` is a custom CUDA op that `torch.export` cannot trace
through anyway — so both problems have the same answer.

Rather than copy ~700 lines and let them drift from upstream, this installs
stand-ins for exactly the names the two model files import and then loads those
files unchanged, by path. The only substantive substitution is the deformable
convolution: `torchvision.ops.deform_conv2d` implements the same modulated
(v2) operator mmcv provides, is a native op, and traces.

    from tools.basicvsrpp import load_basicvsrpp
    model = load_basicvsrpp("gen/basicvsrpp/basicvsr_pp_reds4.pth")
    out = model(lr)          # (N, T, 3, H, W) -> (N, T, 3, 4H, 4W)
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

REPO = Path(__file__).resolve().parent.parent / "dev" / "BasicVSR_PlusPlus"


# ---------------------------------------------------------------- mmcv stand-ins

def _constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding,
                             dilation, groups, deform_groups):
    """mmcv's functional signature, on torchvision's native operator.

    mmcv passes `deform_groups` separately and folds `groups` into the weight
    layout; torchvision infers both from the offset/mask channel counts and the
    weight shape, so they need no translation — only the argument order does.
    """
    return deform_conv2d(x, offset, weight, bias=bias, stride=stride,
                         padding=padding, dilation=dilation, mask=mask)


class ModulatedDeformConv2d(nn.Module):
    """The subset of mmcv's module that `SecondOrderDeformableAlignment` uses.

    It subclasses this and overrides `forward`, so only the constructor's
    parameter registration has to match — the names and shapes are what the
    checkpoint is keyed on.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = int(groups)
        self.deform_groups = int(deform_groups)
        self.with_bias = bool(bias)
        self.weight = nn.Parameter(torch.empty(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, offset, mask):
        return _modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.deform_groups)


class ConvModule(nn.Module):
    """mmcv's conv+norm+activation wrapper, reduced to what SPyNet uses.

    The name `conv` matters: the checkpoint keys are
    `spynet.basic_module.N.basic_module.M.conv.weight`, so the submodule has to
    be attributed exactly this way for `load_state_dict` to match. SPyNet passes
    `norm_cfg=None` throughout, so only the activation is variable.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias="auto",
                 conv_cfg=None, norm_cfg=None, act_cfg=dict(type="ReLU"), **kw):
        super().__init__()
        assert norm_cfg is None, "only the norm-free ConvModule is shimmed"
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=True if bias == "auto" else bool(bias))
        if act_cfg is None:
            self.activate = None
        else:
            t = act_cfg.get("type", "ReLU")
            self.activate = {"ReLU": nn.ReLU(inplace=act_cfg.get("inplace", True)),
                             "LeakyReLU": nn.LeakyReLU(
                                 negative_slope=act_cfg.get("negative_slope", 0.01),
                                 inplace=act_cfg.get("inplace", True))}[t]

    def forward(self, x):
        x = self.conv(x)
        return x if self.activate is None else self.activate(x)


def _install_shims():
    """Put the names the upstream files import into `sys.modules`."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []          # mark as a package, or submodule imports fail
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("mmcv")
    def _kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu",
                      bias=0, distribution="normal"):
        if hasattr(module, "weight") and module.weight is not None:
            fn = nn.init.kaiming_normal_ if distribution == "normal" else nn.init.kaiming_uniform_
            fn(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    mod("mmcv.cnn", constant_init=_constant_init, kaiming_init=_kaiming_init,
        ConvModule=ConvModule)
    mod("mmcv.ops", ModulatedDeformConv2d=ModulatedDeformConv2d,
        modulated_deform_conv2d=_modulated_deform_conv2d)
    # `load_checkpoint` is only reached by `init_weights(pretrained=str)`, which
    # this loader never calls — the state dict goes in directly.
    mod("mmcv.runner", load_checkpoint=lambda *a, **k: None)
    mod("mmcv.utils")
    mod("mmcv.utils.parrots_wrapper", _BatchNorm=nn.modules.batchnorm._BatchNorm,
        _InstanceNorm=nn.modules.instancenorm._InstanceNorm)

    class _Registry:
        def register_module(self, *a, **k):
            return (lambda cls: cls) if not a else a[0]

    mod("mmedit")
    mod("mmedit.models")
    mod("mmedit.models.backbones")
    mod("mmedit.models.backbones.sr_backbones")
    mod("mmedit.models.registry", BACKBONES=_Registry())
    mod("mmedit.utils", get_root_logger=lambda *a, **k: None)


def _load_path(name, relpath, package=None):
    """Load an upstream file under `name`. `package` sets `__package__` so the
    file's relative imports (`from .sr_backbone_utils import ...`) resolve
    against the shim package rather than failing."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    m = importlib.util.module_from_spec(spec)
    if package is not None:
        m.__package__ = package
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_upstream():
    """Import the two upstream model files with the shims in place."""
    _install_shims()
    common = types.ModuleType("mmedit.models.common")
    common.__path__ = []
    sys.modules["mmedit.models.common"] = common
    fw = _load_path("mmedit.models.common.flow_warp",
                    "mmedit/models/common/flow_warp.py", "mmedit.models.common")
    utils = _load_path("mmedit.models.common.sr_backbone_utils",
                       "mmedit/models/common/sr_backbone_utils.py", "mmedit.models.common")
    up = _load_path("mmedit.models.common.upsample",
                    "mmedit/models/common/upsample.py", "mmedit.models.common")
    for src in (fw, utils, up):
        for k in dir(src):
            if not k.startswith("_"):
                setattr(common, k, getattr(src, k))

    net = _load_path("_bvsr_net",
                     "mmedit/models/backbones/sr_backbones/basicvsr_net.py")
    sys.modules["mmedit.models.backbones.sr_backbones.basicvsr_net"] = net
    pp = _load_path("_bvsr_pp",
                    "mmedit/models/backbones/sr_backbones/basicvsr_pp.py")
    return pp


def _strip(sd):
    """Checkpoints store the whole restorer; the backbone is under `generator.`."""
    sd = sd.get("state_dict", sd)
    return {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}


def _report(tag, model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[{tag}] missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5]:
            print("   missing:", k)
        for k in list(unexpected)[:5]:
            print("   unexpected:", k)
    return model


def load_basicvsrpp(ckpt, mid_channels=64, num_blocks=7, is_low_res_input=True,
                    device="cpu"):
    """BasicVSR++, weights loaded, in eval mode.

    Defaults are the REDS4 4x super-resolution model (c64n7). The NTIRE
    compressed-video-enhancement checkpoints are `c128n25` with
    `is_low_res_input=False`, which makes the net 1x: it downsamples by 4 in the
    feature extractor and upsamples by 4 at the end, so it restores rather than
    enlarges. That also makes it cheaper in VRAM than the SR model at equal
    input size, despite being ~4x the parameters, because propagation runs at
    H/4 instead of H.

        load_basicvsrpp(".../decompress_track1.pth", 128, 25, is_low_res_input=False)
    """
    pp = _load_upstream()
    model = pp.BasicVSRPlusPlus(mid_channels=mid_channels, num_blocks=num_blocks,
                                is_low_res_input=is_low_res_input,
                                spynet_pretrained=None)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    return _report("basicvsrpp", model, _strip(sd)).to(device).eval()


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR (Chan et al., CVPR 2022) — BasicVSR behind a cleaning module.

    Reimplemented here rather than checked out, because it is a thin wrapper over
    two classes this repo already provides: `ResidualBlocksWithInputConv` and
    `BasicVSRNet`. Matching upstream attribute names (`image_cleaning`,
    `basicvsr`) is what makes the published checkpoint load.

    The point of it for us: BasicVSR++/REDS4 is trained to invert a clean bicubic
    downsample, so on compressed footage it sharpens the codec's artifacts. This
    one is trained with a real-world degradation pipeline that includes
    compression, and cleans before it propagates.
    """

    def __init__(self, net, mid_channels=64, num_propagation_blocks=20,
                 num_cleaning_blocks=20, dynamic_refine_thres=255):
        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.0
        self.image_cleaning = nn.Sequential(
            net.ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True))
        self.basicvsr = net.BasicVSRNet(mid_channels=mid_channels,
                                        num_blocks=num_propagation_blocks,
                                        spynet_pretrained=None)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        # Upstream cleans up to three times, stopping once the residual it wants
        # to add is small. Empirical in the paper; kept so behaviour matches.
        for _ in range(3):
            lqs = lqs.view(-1, c, h, w)
            residues = self.image_cleaning(lqs)
            lqs = (lqs + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break
        return self.basicvsr(lqs)


def load_realbasicvsr(ckpt, device="cpu"):
    """RealBasicVSR 4x, weights loaded, in eval mode."""
    _load_upstream()
    net = sys.modules["_bvsr_net"]
    model = RealBasicVSRNet(net)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    return _report("realbasicvsr", model, _strip(sd)).to(device).eval()


if __name__ == "__main__":
    m = load_basicvsrpp("gen/basicvsrpp/basicvsr_pp_reds4.pth")
    n = sum(p.numel() for p in m.parameters())
    print(f"BasicVSR++ loaded: {n/1e6:.2f}M parameters")
    x = torch.randn(1, 5, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    print("forward:", tuple(x.shape), "->", tuple(y.shape))
