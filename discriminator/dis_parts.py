from torch import nn

class res_unit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_unit, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.bn(x)) + self.conv1(self.bn(x))

class res_unit_nb(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_unit_nb, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x) + self.conv1(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(2, 2),
            res_unit(in_ch, out_ch)
        )
    def forward(self, x):
        return self.conv(x)