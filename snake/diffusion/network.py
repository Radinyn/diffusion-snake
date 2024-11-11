import torch


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding="same", bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding="same", bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Down(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Up(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Fix odd shapes with padding
        h_padding = identity.shape[2] - x.shape[2]
        w_padding = identity.shape[3] - x.shape[3]

        x = torch.nn.functional.pad(
            x,
            [
                w_padding // 2,
                w_padding - w_padding // 2,
                h_padding // 2,
                h_padding - h_padding // 2,
            ],
        )

        x = torch.cat([identity, x], dim=1)
        return self.block(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        noise_frame = 1
        past_frames = 3
        action = 1
        step = 1

        total_channels = noise_frame * 3 + past_frames * 3 + action + step

        self.start = BasicBlock(total_channels, 64)  # 15
        self.down0 = Down(64, 128)  # 7
        self.down1 = Down(128, 256)  # 3
        self.down2 = Down(256, 256)  # 1
        self.up0 = Up(512, 128)  # 3
        self.up1 = Up(256, 64)  # 7
        self.up2 = Up(128, 32)  # 15
        self.end_conv = torch.nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.start(x)  # 15
        x1 = self.down0(x0)  # 7
        x2 = self.down1(x1)  # 3
        x3 = self.down2(x2)  # 1
        x = self.up0(x3, x2)  # 3
        x = self.up1(x, x1)  # 7
        x = self.up2(x, x0)  # 15
        return self.end_conv(x)
