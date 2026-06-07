import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import os

# ==========================================
# 1. Dynamic Learnable Gabor Convolution Layer (Preserve Advantages)
# ==========================================
class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio
        self.kernel = None

        if init_ratio <= 0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

        gb = gb - gb.mean(dim=[2,3], keepdim=True)
        return gb

    def forward(self, x):
        if self.training or self.kernel is None:
            kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out,
                                    self.sigma, self.gamma, self.theta, self.f, self.psi)
            self.kernel = kernel
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

# ==========================================
# 2. Coordinate Attention Mechanism CoordAtt (Core Upgrade: Replaces SE)
# ==========================================
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Coordinate weighting, forcing the network to remember the horizontal and vertical orientation of palmprints
        out = identity * a_w * a_h
        return out

# ==========================================
# 3. Coordinate-Aware Competitive Block
# ==========================================
class CoordinateAwareCompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding, weight, init_ratio=1, o1=32, o2=12):
        super(CoordinateAwareCompetitiveBlock, self).__init__()

        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2, padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2, padding=ksize // 2, init_ratio=init_ratio)

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)

        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        # [Core Replacement]: The original SELayer has been replaced with CoordAtt capable of capturing spatial topology
        self.ca1 = CoordAtt(n_competitor, n_competitor)
        self.ca2 = CoordAtt(n_competitor, n_competitor)

        if weight is None:
            weight = 1.0
        self.weight_chan = weight
        self.weight_spa = (1 - weight) / 2

    def forward(self, x):
        # 1-st order
        x = self.gabor_conv2d(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        # Spatial topology coordinate attention weighting
        x_1 = self.ca1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        # 2-nd order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)

        # Spatial topology coordinate attention weighting
        x_2 = self.ca2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0], -1), x_2.view(x_2.shape[0], -1)), dim=1)
        return xx

# ==========================================
# 4. ArcFace Angular Classification Head (Preserve Advantages)
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-6, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine
        return output

# ==========================================
# 5. Complete Co3Net Backbone Network (Officially Named)
# ==========================================
class co3net(torch.nn.Module):
    '''
    Co3Net = Coordinate-Aware Competitive Block * 3 + FC + ArcFace Output
    '''
    def __init__(self, num_classes, weight=None):
        super(co3net, self).__init__()
        self.num_classes = num_classes

        # Use the new competitive blocks injected with CoordAtt
        self.cb1 = CoordinateAwareCompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
        self.cb2 = CoordinateAwareCompetitiveBlock(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CoordinateAwareCompetitiveBlock(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)

        # Fully-connected layer dimensions are fully compatible, no changes needed
        self.fc = torch.nn.Linear(13152, 4096)
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def load_pretrained_weights(self, path):
        if os.path.exists(path):
            pretrained_dict = torch.load(path)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and "arclayer_" not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded pre-trained weights from {path}")
        else:
            print("No pre-trained weights found, training from scratch.")

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)
        x = self.drop(x)
        x = self.arclayer_(x, y)

        # Return classification logits and the expressive latent feature for SupCon and knowledge distillation
        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

if __name__== "__main__" :
    # Dimension check: ensure dimensions are correct and no memory overflow occurs
    inp = torch.randn(2, 1, 128, 128)
    net = co3net(600, weight=0.8)
    out, feat = net(inp, torch.tensor([0, 1]))
    print(f"Output Logits Shape: {out.shape}")  # Expected: [2, 600]
    print(f"Latent Feature Shape: {feat.shape}") # Expected: [2, 6144]
    print("Co3Net Initialized Successfully!")
