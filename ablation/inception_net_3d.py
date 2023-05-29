import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(InceptionBlock, self).__init__()
        # We ensure that the only the convolution layer with stride 2 causes a downsampling.
        # Thus, we are ensuring that all the shapes are same except for channel dimension.
        # We can thus, concatenate the outputs easily.
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(),
            nn.Conv3d(in_channels=out_channels // 4, out_channels=out_channels // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout3d(),
            nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout3d()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(),
            nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout3d()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout3d()
        )

        self.mp = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout3d()
        )

    def forward(self, x):
        out = [self.conv1(x), self.conv2(x), self.conv3(x), self.mp(x)]
        return torch.cat(out, dim=1)


class InceptionModel(nn.Module):
    def __init__(self, in_channel, block1_out_ch, block2_out_ch, block3_out_ch, vol_size, num_classes=2):
        super(InceptionModel, self).__init__()
        self.block1 = InceptionBlock(in_channel=in_channel, out_channels=block1_out_ch)
        self.block2 = InceptionBlock(in_channel=(3 * block1_out_ch + in_channel), out_channels=block2_out_ch)
        self.block3 = InceptionBlock(in_channel=(3 * (block2_out_ch + block1_out_ch) + in_channel),
                                     out_channels=block3_out_ch)
        # We use three inception blocks leading to decrease in volume by a factor of 8.
        volume_dim_final = vol_size // (2 ** 3)
        final_out_ch = 3 * (block3_out_ch + block2_out_ch + block1_out_ch) + in_channel
        self.linear = nn.Sequential(
            nn.Linear((volume_dim_final ** 3) * final_out_ch, block3_out_ch),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(block3_out_ch, num_classes)
        self.regr = nn.Linear(block3_out_ch, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        clf = self.fc1(x)
        regr_out = self.regr(x)
        regr_out = torch.log(1 + torch.exp(regr_out - regr_out.max())) + regr_out.max()
        return clf, regr_out


def weight_reset(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == '__main__':
    model = InceptionModel(in_channel=2, block1_out_ch=16, block2_out_ch=32, block3_out_ch=64, vol_size=144)
    x = torch.randn(2, 2, 144, 144, 144)
    print(model(x)[0].shape)
    model.apply(weight_reset)
