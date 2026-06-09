"""
Beta-encoder issu de DIDT-CLIP
utilisé en amont pour précalculer l'anatomie ?


Contenu :
  - ConvBlock2d, Upsample, UNet  (architecture de DIST-CLIP)
  - load_beta_encoder()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path



class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return torch.cat([skip, x], dim=1)


class Beta_encoder(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act="noact", use_contrast_norm=False):
        super().__init__()
        self.final_act        = final_act
        self.use_contrast_norm = use_contrast_norm
        self.in_conv       = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.contrast_norm = nn.InstanceNorm2d(in_ch, affine=True)

        self.down_convs   = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples   = nn.ModuleList()
        self.up_convs     = nn.ModuleList()

        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(2, 2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))

        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x, condition=None):
        if self.use_contrast_norm:
            x = self.contrast_norm(x)

        encoded_features = []
        x = self.in_conv(x)

        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feat_dim = x.shape[-1]
                x = down_conv(
                    torch.cat([x, condition.repeat(1, 1, feat_dim, feat_dim)], dim=1)
                )
            else:
                x = down_conv(x)
            encoded_features.append(x)
            x = down_sample(x)

        x = self.bottleneck_conv(x)

        for skip, up_sample, up_conv in zip(
            reversed(encoded_features),
            reversed(self.up_samples),
            reversed(self.up_convs),
        ):
            x = up_sample(x, skip)
            x = up_conv(x)

        x = self.out_conv(x)

        if self.final_act == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == "tanh":
            x = torch.tanh(x)
        else:
            # 'noact' : pas d'activation
            x = x 

        return x




def load_beta_encoder(checkpoint_path: Path, device: torch.device) -> Beta_encoder:
    checkpoint_path = Path(checkpoint_path)
    reloaded    = torch.load(checkpoint_path, map_location="cpu")
    reloaded_sd = reloaded["state_dict"]

    in_ch   = reloaded_sd["in_conv.weight"].shape[1]
    base_ch = reloaded_sd["in_conv.weight"].shape[0]
    out_ch  = reloaded_sd["out_conv.2.weight"].shape[0]
    num_lvs = sum(
        1 for k in reloaded_sd
        if k.startswith("down_convs.") and k.endswith(".conv.0.weight")
    )

    print("Hyperparamètres détectés depuis le checkpoint :")
    print(f"  in_ch={in_ch}  out_ch={out_ch}  base_ch={base_ch}  num_lvs={num_lvs}")

    model = Beta_encoder(
        in_ch=in_ch,
        out_ch=out_ch,
        base_ch=base_ch,
        num_lvs=num_lvs,
        final_act="noact",
        use_contrast_norm=True,
    )
    model.load_state_dict(reloaded_sd, strict=True)
    model.eval()
    model.to(device)

    print(f"  Chargement strict=True OK — modèle sur {device}")
    return model
