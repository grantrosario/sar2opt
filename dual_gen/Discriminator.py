import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
from torchvision.transforms import transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class DiscriminatorBranch(nn.Module):
    def __init__(self, in_channels):
        super(DiscriminatorBranch, self).__init__()
        self.residual_block = ResidualBlock(in_channels)
        self.conv = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        edge_map = self.residual_block(x)
        features = self.relu(self.conv(edge_map))
        return features

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.structure_branch = DiscriminatorBranch(1)
        self.texture_branch = DiscriminatorBranch(3)
        self.final_conv = spectral_norm(nn.Conv2d(5, 1, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, edges):

        # grayscale features
        grays = transforms.Grayscale(num_output_channels=1)(images)

        # Structure branch for real and pseudo images
        structure_features = self.structure_branch(edges)
        structure_features = torch.cat((structure_features, grays), 1)
        texture_features = self.texture_branch(images)
        combined_features = torch.cat((structure_features, texture_features), 1)
        logits = self.final_conv(combined_features)
        probs = self.sigmoid(logits)

        return probs

        # # create labels for loss
        # batch_size = real_image.size(0)
        # real_labels = torch.ones(batch_size, 1)
        # pseudo_labels = torch.zeros(batch_size, 1)

        # if real_image.is_cuda:
        #     real_labels = real_labels.cuda()
        #     pseudo_labels = pseudo_labels.cuda()


