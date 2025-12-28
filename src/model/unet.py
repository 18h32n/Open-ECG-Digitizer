from typing import List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class UNet(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        depth: int,
        dims: List[int],
        use_checkpoint: bool = False,
    ):
        super(UNet, self).__init__()

        self.depth = depth
        self.dims = dims
        self.use_checkpoint = use_checkpoint

        # Encoder blocks
        self.encoders = nn.ModuleList(
            [
                self._make_encoder_block(num_in_channels if i == 0 else dims[i - 1], dims[i], depth)
                for i in range(len(dims))
            ]
        )
        self.encoder_skips = nn.ModuleList(
            [self._make_skip_connection(num_in_channels if i == 0 else dims[i - 1], dims[i]) for i in range(len(dims))]
        )
        self.encoder_downscaling = nn.ModuleList(
            [
                nn.Conv2d(
                    dims[i],
                    dims[i],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
                for i in range(len(dims))
            ]
        )

        # Decoder blocks
        self.decoders = nn.ModuleList(
            [self._make_decoder_block(dims[i] + dims[i - 1], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )
        self.decoder_skips = nn.ModuleList(
            [self._make_skip_connection(dims[i] + dims[i - 1], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )

        # Final output layer
        self.final_conv = nn.Conv2d(
            dims[0], num_out_channels, kernel_size=3, padding=1, bias=False, padding_mode="replicate"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Forward through encoders
        for encoder, skip, down in zip(self.encoders, self.encoder_skips, self.encoder_downscaling):
            x = encoder(x) + skip(x)  # Add skip connection
            skips.append(x)
            x = down(x)  # Downscale

        # Reverse skips for decoding
        skips = skips[::-1]
        x = skips[0]  # Start decoding from the last skip

        # Forward through decoders
        for i, (decoder, skip) in enumerate(zip(self.decoders, self.decoder_skips)):
            x = self._upsample(x, skips[i + 1])
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory during training
                x = checkpoint(decoder, torch.cat([x, skips[i + 1]], dim=1), use_reentrant=False)
            else:
                x = decoder(torch.cat([x, skips[i + 1]], dim=1))

        # Final convolution to match output channels
        x = self.final_conv(x)
        return x

    def _make_encoder_block(self, in_channels: int, num_out_channels: int, num_layers: int) -> nn.Sequential:
        layers = [self._conv_norm_act(in_channels, num_out_channels)]
        for _ in range(num_layers - 1):
            layers.append(self._conv_norm_act(num_out_channels, num_out_channels))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self._conv_norm_act(in_channels, num_out_channels),
            self._conv_norm_act(num_out_channels, num_out_channels),
        )

    def _make_skip_connection(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_out_channels,
                kernel_size=1,
                bias=False,
                padding_mode="replicate",
            ),
        )

    def _conv_norm_act(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.InstanceNorm2d(num_out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def _upsample(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return x
