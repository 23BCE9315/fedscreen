
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import (
    EfficientNet_B0_Weights
)


# ==============================================================================
# BatchNorm -> GroupNorm
# ==============================================================================

def replace_bn_with_gn(
    module: nn.Module,
    num_groups: int = 32
):

    for name, child in module.named_children():

        if isinstance(
            child,
            nn.BatchNorm2d
        ):

            num_channels = (
                child.num_features
            )

            g = min(
                num_groups,
                num_channels
            )

            while (
                g > 1
                and num_channels % g != 0
            ):
                g //= 2

            gn = nn.GroupNorm(
                num_groups=g,
                num_channels=num_channels
            )

            setattr(
                module,
                name,
                gn
            )

        else:

            replace_bn_with_gn(
                child,
                num_groups
            )

    return module


# ==============================================================================
# Channel Attention
# ==============================================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction_ratio=16
    ):

        super().__init__()

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        hidden_channels = max(
            8,
            in_channels // reduction_ratio
        )

        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                in_channels,
                1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_out + max_out
        )

        return x * attention


# ==============================================================================
# Spatial Attention
# ==============================================================================

class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = (
            3
            if kernel_size == 7
            else 1
        )

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [
                avg_out,
                max_out
            ],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ==============================================================================
# CBAM Block
# ==============================================================================

class CBAMBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction_ratio=16
    ):

        super().__init__()

        self.channel_att = (
            ChannelAttention(
                in_channels,
                reduction_ratio
            )
        )

        self.spatial_att = (
            SpatialAttention(
                kernel_size=7
            )
        )

    def forward(self, x):

        x = self.channel_att(x)
        x = self.spatial_att(x)

        return x


# ==============================================================================
# Disease Task Head
# ==============================================================================

class DiseaseTaskHead(nn.Module):

    def __init__(
        self,
        in_channels=1280,
        num_classes=2,
        dropout_rate=0.3
    ):

        super().__init__()

        self.cbam = CBAMBlock(
            in_channels=in_channels,
            reduction_ratio=16
        )

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                p=dropout_rate
            ),

            nn.Linear(
                in_channels,
                256
            ),

            nn.GroupNorm(
                32,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=dropout_rate / 2.0
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.cbam(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            1
        )

        return self.classifier(x)


# ==============================================================================
# Complete FedScreen Model
# ==============================================================================

class FedScreenMultiTaskModel(nn.Module):

    def __init__(
        self,
        num_classes_malaria=2,
        num_classes_anemia=2,
        num_classes_leukemia=2
    ):

        super().__init__()

        # The trained FedProx checkpoint already contains the complete
        # EfficientNet-B0 backbone weights. Do not download ImageNet
        # weights at runtime; this makes local/cloud deployment reliable.
        base_model = models.efficientnet_b0(weights=None)

        self.backbone = (
            base_model.features
        )

        self.backbone = (
            replace_bn_with_gn(
                self.backbone,
                num_groups=32
            )
        )

        in_channels = 1280

        self.malaria_head = (
            DiseaseTaskHead(
                in_channels,
                num_classes_malaria
            )
        )

        self.anemia_head = (
            DiseaseTaskHead(
                in_channels,
                num_classes_anemia
            )
        )

        self.leukemia_head = (
            DiseaseTaskHead(
                in_channels,
                num_classes_leukemia
            )
        )

    def forward(self, x):

        shared_features = (
            self.backbone(x)
        )

        return {
            "malaria": self.malaria_head(
                shared_features
            ),

            "anemia": self.anemia_head(
                shared_features
            ),

            "leukemia": self.leukemia_head(
                shared_features
            )
        }
