# models/csrnet.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

# choices for "activation" arg
ACTIVATIONS = ("none", "relu", "softplus")


# TODO: better comments
class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True, activation="softplus"):
        """
        args:
            load_pretrained: use imagenet-pretrained vgg16 weights for the frontend
            activation: how negative density values are handled at the output
                "none" - no clamping
                "relu" - F.relu(x)
                "softplus" - F.softplus(x)
        """
        super(CSRNet, self).__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {activation!r}"
            )
        self.activation = activation

        if load_pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
        else:
            weights = None

        # csr net frontend: vgg16 convolution layers
        vgg = models.vgg16(weights=weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # backend: dilated convolutions to ..........
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)

        # negative value handling, chosen at init
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "softplus":
            x = F.softplus(x)
        # else no negative clapming

        return x
